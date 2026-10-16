import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

import comfy.audio
import comfy.model_management

from .index import blend_index_features

# Upstream removes DC and rumble with a 5th-order 48 Hz Butterworth high-pass, run forward and backward.
_HP_B, _HP_A = signal.butter(N=5, Wn=48, btype="high", fs=16000)
_F0_MEL_MIN = 1127 * math.log(1 + 50 / 700)
_F0_MEL_MAX = 1127 * math.log(1 + 1100 / 700)


def highpass_48hz(x):
    return torch.from_numpy(signal.filtfilt(_HP_B, _HP_A, x.double().numpy()).copy()).float()


def reflect_pad(x, pad):
    """Reflect-pad a 1-D signal like upstream's np.pad, which (unlike torch) also handles pads longer than the input."""
    return torch.from_numpy(np.pad(x.numpy(), pad, mode="reflect"))


def coarse_f0(f0_hz, f0_up_key=0):
    """Shifted f0 in Hz and its 1..255 coarse mel bins. Like upstream, unvoiced frames are
    linearly interpolated from the voiced ones first (left as 0 Hz only when nothing is voiced)."""
    f0 = f0_hz.float().clone()
    unvoiced = f0 == 0
    if not unvoiced.all():
        idx = torch.arange(f0.shape[0])
        f0[unvoiced] = torch.from_numpy(np.interp(idx[unvoiced].numpy(), idx[~unvoiced].numpy(), f0[~unvoiced].numpy())).float()
    f0 = f0 * 2 ** (f0_up_key / 12)
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - _F0_MEL_MIN) * 254 / (_F0_MEL_MAX - _F0_MEL_MIN) + 1
    return f0_mel.clamp(1, 255).round().long(), f0


def change_rms(data1, sr1, data2, sr2, rate):
    def rms(y, frame, hop):
        yp = F.pad(y.unsqueeze(0), (frame // 2, frame // 2)).squeeze(0)
        return yp.unfold(0, frame, hop).pow(2).mean(-1).sqrt()
    rms1 = rms(data1, sr1 // 2 * 2, sr1 // 2)
    rms2 = rms(data2, sr2 // 2 * 2, sr2 // 2)
    rms1 = F.interpolate(rms1[None, None], size=data2.shape[0], mode="linear").squeeze()
    rms2 = F.interpolate(rms2[None, None], size=data2.shape[0], mode="linear").squeeze().clamp_min(1e-6)
    return data2 * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1))


class RVCPipeline:
    def __init__(self, tgt_sr, x_pad=1, x_query=6, x_center=38, x_max=41):
        self.tgt_sr = tgt_sr
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * x_pad
        self.t_pad_tgt = tgt_sr * x_pad
        self.t_query = self.sr * x_query
        self.t_center = self.sr * x_center
        self.t_max = self.sr * x_max

    def vc(self, hubert, net_g, sid, audio0, pitch, pitchf, index_vectors, index_rate, version, protect, generator):
        device = sid.device
        states = hubert(audio0.view(1, -1).to(device, hubert.dtype))
        # v1 voices were trained on the 256-dim projection of layer 9, v2 on raw layer 12.
        feats = hubert.final_proj(states[8]) if version == "v1" else states[11]
        use_protect = protect < 0.5 and pitch is not None
        if use_protect:
            feats0 = feats
        if index_vectors is not None and index_rate != 0:
            feats = blend_index_features(feats[0], index_vectors, index_rate).unsqueeze(0)
        feats = F.interpolate(feats.transpose(1, 2), scale_factor=2).transpose(1, 2)
        p_len = min(audio0.shape[0] // self.window, feats.shape[1])
        feats = feats[:, :p_len]
        if pitch is not None:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]
        if use_protect:
            feats0 = F.interpolate(feats0.transpose(1, 2), scale_factor=2).transpose(1, 2)[:, :p_len]
            # Unvoiced frames keep `protect` of the indexed features, voiced frames all of them.
            mix = torch.where(pitchf > 0, 1.0, protect).unsqueeze(-1).to(feats.dtype)
            feats = torch.lerp(feats0, feats, mix)
        lengths = torch.tensor([p_len], device=device)
        return net_g.infer(feats, lengths, sid, generator, pitch, pitchf)[0, 0].float().cpu()

    def split_points(self, audio):
        """Cut points near every t_center samples, at the quietest 10 ms of the surrounding t_query window."""
        if audio.shape[0] + self.window <= self.t_max:
            return []
        audio_pad = reflect_pad(audio, self.window // 2)
        energy = torch.cat((torch.zeros(1, dtype=torch.float64), audio_pad.abs().double().cumsum(0)))
        audio_sum = (energy[self.window:self.window + audio.shape[0]] - energy[:audio.shape[0]]).float()
        opt_ts = []
        for t in range(self.t_center, audio.shape[0], self.t_center):
            seg = audio_sum[t - self.t_query:t + self.t_query]
            opt_ts.append(t - self.t_query + int(seg.argmin()))
        return opt_ts

    def convert(self, hubert, net_g, sid, audio, pitch_models, f0_up_key, index_vectors,
                index_rate, version, protect, rms_mix_rate, resample_sr, device, generator):
        """pitch_models: [] for voices without f0, else one or two pitch estimators (two are averaged where both are voiced)."""
        audio = highpass_48hz(audio)
        opt_ts = self.split_points(audio)
        audio_pad = reflect_pad(audio, self.t_pad)
        p_len = audio_pad.shape[0] // self.window
        sid_t = torch.tensor([sid], device=device)
        pitch, pitchf = None, None
        if pitch_models:
            # Upstream estimates f0 on the high-passed, padded signal.
            f0s = [m(audio_pad[None].to(device))[0].cpu() for m in pitch_models]
            f0_hz = f0s[0]
            if len(f0s) == 2:
                a, b = f0s
                f0_hz = torch.where((a > 0) & (b > 0), (a + b) / 2, torch.maximum(a, b))
            pitch, pitchf = coarse_f0(f0_hz[:p_len], f0_up_key)
            pitch = pitch.to(device).unsqueeze(0)
            pitchf = pitchf.to(device).unsqueeze(0)
        if index_vectors is not None and index_rate != 0:
            index_vectors = index_vectors.to(device)
        bounds = [t // self.window * self.window for t in opt_ts]
        starts = [0] + bounds
        ends = [t + self.t_pad * 2 + self.window for t in bounds] + [audio_pad.shape[0]]
        audio_opt = []
        for s, e in zip(starts, ends):
            comfy.model_management.throw_exception_if_processing_interrupted()
            sl = slice(s // self.window, e // self.window)
            seg = self.vc(hubert, net_g, sid_t, audio_pad[s:e],
                          pitch[:, sl] if pitch is not None else None,
                          pitchf[:, sl] if pitchf is not None else None,
                          index_vectors, index_rate, version, protect, generator)
            audio_opt.append(seg[self.t_pad_tgt:-self.t_pad_tgt])
        out = torch.cat(audio_opt)
        if rms_mix_rate != 1:
            out = change_rms(audio, 16000, out, self.tgt_sr, rms_mix_rate)
        sr = self.tgt_sr
        if self.tgt_sr != resample_sr >= 16000:
            out = comfy.audio.resample(out[None, None], self.tgt_sr, resample_sr)[0, 0]
            sr = resample_sr
        peak = out.abs().max() / 0.99
        if peak > 1:
            out = out / peak
        return out, sr

