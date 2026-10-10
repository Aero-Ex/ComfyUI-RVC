import numpy as np
import torch
import torch.nn.functional as F

from .index import blend_index_features

# Steady-state delay vector for the filter above (scipy.signal.lfilter_zi),
# zi = _HP_ZI * x0. Baked so shipped code needs no scipy.
_HP_ZI = [-0.9699607413707367, 3.879842959615721, -5.81976443080129,
          3.879842948235015, -0.9699607356787477]
_HP_B = [0.9699606451838447, -4.849803225919223, 9.699606451838447,
         -9.699606451838447, 4.849803225919223, -0.9699606451838447]
_HP_A = [1.0, -4.939001819168364, 9.757863526739543, -9.639544849413458,
         4.761506797356209, -0.9408236532054606]


def _lfilter(b, a, x, n):
    bp = torch.nn.functional.pad(b, (0, n + 1 - b.shape[0]))
    ap = torch.nn.functional.pad(a, (0, n + 1 - a.shape[0]))
    w = torch.zeros(n, dtype=torch.float64)
    y = torch.empty(x.shape[0], dtype=torch.float64)
    for i, xn in enumerate(x):
        yn = bp[0] * xn + w[0]
        w[:-1] = w[1:] + bp[1:-1] * xn - ap[1:-1] * yn
        w[-1] = bp[-1] * xn - ap[-1] * yn
        y[i] = yn
    return y, w


def highpass_48hz(x):
    b = torch.tensor(_HP_B, dtype=torch.float64)
    a = torch.tensor(_HP_A, dtype=torch.float64)
    n = max(b.shape[0], a.shape[0]) - 1
    xd = x.double()
    padlen = 18
    left = 2 * xd[0] - xd[1:padlen + 1].flip(0)
    right = 2 * xd[-1] - xd[-padlen - 1:-1].flip(0)
    ext = torch.cat([left, xd, right])
    zi = torch.tensor(_HP_ZI, dtype=torch.float64)
    y, _ = _lfilter_zi(b, a, ext, zi * ext[0], n)
    y2, _ = _lfilter_zi(b, a, y.flip(0), zi * y[-1], n)
    n0 = xd.shape[0]
    return y2.flip(0)[padlen:padlen + n0].float()


def _lfilter_zi(b, a, x, w_init, n):
    bp = torch.nn.functional.pad(b, (0, n + 1 - b.shape[0]))
    ap = torch.nn.functional.pad(a, (0, n + 1 - a.shape[0]))
    if w_init is None:
        _, w_init = _lfilter(b, a, torch.full((10000,), x[0].item(), dtype=torch.float64), n)
    w = w_init.clone()
    y = torch.empty(x.shape[0], dtype=torch.float64)
    for i, xn in enumerate(x):
        yn = bp[0] * xn + w[0]
        w[:-1] = w[1:] + bp[1:-1] * xn - ap[1:-1] * yn
        w[-1] = bp[-1] * xn - ap[-1] * yn
        y[i] = yn
    return y, w


def coarse_f0(f0_hz, f0_up_key=0):
    f0 = f0_hz.clone() if isinstance(f0_hz, torch.Tensor) else torch.tensor(f0_hz)
    f0 = f0.float()
    voiced = f0 != 0
    if bool(voiced.any()) and bool((~voiced).any()):
        idx = torch.arange(f0.shape[0])
        f0[~voiced] = torch.from_numpy(np.interp(idx[~voiced].numpy(), idx[voiced].numpy(), f0[voiced].numpy())).float()
    f0 = f0 * pow(2, f0_up_key / 12)
    f0bak = f0.clone()
    f0_mel = 1127 * torch.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)
    pos = f0_mel > 0
    f0_mel[pos] = (f0_mel[pos] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255
    return f0_mel.round().int(), f0bak


def change_rms(data1, sr1, data2, sr2, rate):
    def rms(y, frame, hop):
        yp = F.pad(y.unsqueeze(0), (frame // 2, frame // 2)).squeeze(0)
        win = yp.unfold(0, frame, hop)
        return win.pow(2).mean(-1).sqrt()
    rms1 = rms(data1, sr1 // 2 * 2, sr1 // 2)
    rms2 = rms(data2, sr2 // 2 * 2, sr2 // 2)
    rms1 = F.interpolate(rms1.unsqueeze(0).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = F.interpolate(rms2.unsqueeze(0).unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.clamp(rms2, min=1e-6)
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

    def extract_features(self, hubert, wav, version, device):
        dtype = getattr(hubert, "dtype", None) or wav.dtype
        feats = wav.view(1, -1).to(device=device, dtype=dtype)
        with torch.inference_mode():
            states = hubert(feats)
        return states[8] if version == "v1" else states[11]

    def vc(self, hubert, net_g, sid, audio0, pitch, pitchf, index_vectors, index_rate, version, protect, device):
        feats = self.extract_features(hubert, audio0, version, device).to(device)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if index_vectors is not None and index_rate != 0:
            feats = blend_index_features(feats[0].float().cpu(), index_vectors, index_rate).unsqueeze(0).to(device, feats.dtype)
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len_t = torch.tensor([p_len], device=device).long()
        with torch.inference_mode():
            if pitch is not None and pitchf is not None:
                audio1 = net_g.infer(feats, p_len_t, pitch, pitchf, sid)[0]
            else:
                audio1 = net_g.infer(feats, p_len_t, sid)[0]
        return audio1[0, 0].float().cpu()

    def split_points(self, audio):
        audio_pad = F.pad(audio.unsqueeze(0), (self.window // 2, self.window // 2), mode="reflect").squeeze(0)
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = torch.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i:i + self.window].abs()
            for t in range(self.t_center, audio.shape[0], self.t_center):
                seg = audio_sum[t - self.t_query:t + self.t_query]
                opt_ts.append(t - self.t_query + int(seg.argmin().item()))
        return opt_ts

    def convert(self, hubert, net_g, sid, audio, f0_hz, f0_up_key, if_f0, index_vectors,
                index_rate, version, protect, rms_mix_rate, resample_sr, resample_fn, device):
        audio = highpass_48hz(audio)
        opt_ts = self.split_points(audio)
        audio_pad = F.pad(audio.unsqueeze(0), (self.t_pad, self.t_pad), mode="reflect").squeeze(0)
        p_len = audio_pad.shape[0] // self.window
        sid_t = torch.tensor(sid, device=device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = coarse_f0(f0_hz[:p_len] if f0_hz.shape[0] >= p_len else f0_hz, f0_up_key)
            pitch = pitch[:p_len].to(device).unsqueeze(0).long()
            pitchf = pitchf[:p_len].float().to(device).unsqueeze(0).float()
        audio_opt = []
        s = 0
        t = None
        for t in opt_ts:
            t = t // self.window * self.window
            e = t + self.t_pad * 2 + self.window
            sl = slice(s // self.window, e // self.window)
            seg = self.vc(hubert, net_g, sid_t, audio_pad[s:e],
                          pitch[:, sl] if pitch is not None else None,
                          pitchf[:, sl] if pitchf is not None else None,
                          index_vectors, index_rate, version, protect, device)
            audio_opt.append(seg[self.t_pad_tgt:-self.t_pad_tgt])
            s = t
        sl = slice((t // self.window if t is not None else 0), None)
        seg = self.vc(hubert, net_g, sid_t, audio_pad[t:] if t is not None else audio_pad,
                      pitch[:, sl] if pitch is not None else None,
                      pitchf[:, sl] if pitchf is not None else None,
                      index_vectors, index_rate, version, protect, device)
        audio_opt.append(seg[self.t_pad_tgt:-self.t_pad_tgt])
        out = torch.cat(audio_opt)
        if rms_mix_rate != 1:
            out = change_rms(audio, 16000, out, self.tgt_sr, rms_mix_rate)
        if self.tgt_sr != resample_sr >= 16000:
            out = resample_fn(out, self.tgt_sr, resample_sr)
            sr = resample_sr
        else:
            sr = self.tgt_sr
        peak = out.abs().max() / 0.99
        if peak > 1:
            out = out / peak
        return out, sr
