import json

import folder_paths
import comfy.audio
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import torch

from .rvc.hubert import HubertEncoder
from .rvc.rmvpe import RMVPEModel, log_mel_spectrogram
from .rvc.fcpe import FCPitch
from .rvc.index import load_index_vectors
from .rvc.pipeline import RVCPipeline, highpass_48hz
from .rvc import synth as rvc_synth


def model_dtype(model):
    return getattr(model, "dtype", None) or torch.float32


SYNTH_CLASSES = {
    ("v1", 1): rvc_synth.SynthesizerTrnMs256NSFsid,
    ("v1", 0): rvc_synth.SynthesizerTrnMs256NSFsid_nono,
    ("v2", 1): rvc_synth.SynthesizerTrnMs768NSFsid,
    ("v2", 0): rvc_synth.SynthesizerTrnMs768NSFsid_nono,
}


def _make_operations(sd):
    dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32])
    manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, comfy.model_management.get_torch_device(), supported_dtypes=[torch.float16, torch.bfloat16, torch.float32])
    quant = comfy.utils.detect_layer_quantization(sd, "")
    if quant is not None:
        return comfy.ops.mixed_precision_ops(quant, dtype), dtype
    return comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=comfy.model_management.get_torch_device()), dtype


def _wrap(model):
    return comfy.model_patcher.ModelPatcher(
        model, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())


class RVCLoaderHubert:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rvc_name": (folder_paths.get_filename_list("rvc"), {
                "tooltip": "Converted RVC hubert_base.safetensors."}),
        }}

    RETURN_TYPES = ("RVC_HUBERT",)
    RETURN_NAMES = ("hubert",)
    FUNCTION = "execute"
    CATEGORY = "RVC"

    def execute(self, rvc_name):
        model_path = folder_paths.get_full_path_or_raise("rvc", rvc_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
        metadata = metadata or {}
        if metadata.get("rvc_hubert.format_version") != "1" or "post_extract_proj.weight" not in sd:
            raise RuntimeError("ERROR: not a converted RVC Hubert checkpoint: %s" % rvc_name)
        operations, dtype = _make_operations(sd)
        model = HubertEncoder(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=operations)
        model.load_state_dict(sd, strict=True)
        model.dtype = dtype
        return (_wrap(model),)


class RVCLoaderPitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rvc_name": (folder_paths.get_filename_list("rvc"), {
                "tooltip": "Converted rmvpe.safetensors or fcpe.safetensors."}),
        }}

    RETURN_TYPES = ("RVC_PITCH",)
    RETURN_NAMES = ("pitch",)
    FUNCTION = "execute"
    CATEGORY = "RVC"

    def execute(self, rvc_name):
        model_path = folder_paths.get_full_path_or_raise("rvc", rvc_name)
        sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
        metadata = metadata or {}
        operations, dtype = _make_operations(sd)
        if metadata.get("rvc_rmvpe.format_version") == "1" and "fc.1.weight" in sd:
            model = RMVPEModel(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=operations)
            model.mel_basis = sd.pop("rvc.mel_basis").to(dtype=torch.float32)
            model.load_state_dict(sd, strict=True)
            model.pitch_kind = "rmvpe"
            model.dtype = dtype
        elif metadata.get("rvc_fcpe.format_version") == "1" and "input_stack.0.weight" in sd:
            model = FCPitch(device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=operations)
            model.load_state_dict(sd, strict=True)
            model.pitch_kind = "fcpe"
            model.dtype = dtype
        else:
            raise RuntimeError("ERROR: not a converted RVC pitch checkpoint: %s" % rvc_name)
        return (_wrap(model),)


class RVCLoaderVoice:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rvc_name": (folder_paths.get_filename_list("rvc"), {
                "tooltip": "RVC voice .pth model."}),
        }}

    RETURN_TYPES = ("RVC_VOICE",)
    RETURN_NAMES = ("voice",)
    FUNCTION = "execute"
    CATEGORY = "RVC"

    def execute(self, rvc_name):
        model_path = folder_paths.get_full_path_or_raise("rvc", rvc_name)
        cpt = torch.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(cpt, dict) or "weight" not in cpt or "config" not in cpt or len(cpt["config"]) != 18:
            raise RuntimeError("ERROR: not an RVC voice checkpoint: %s" % rvc_name)
        version = cpt.get("version", "v1")
        if_f0 = int(cpt.get("f0", 1))
        cls = SYNTH_CLASSES.get((version, if_f0))
        if cls is None:
            raise RuntimeError("ERROR: unknown RVC voice version/f0: %s/%s" % (version, if_f0))
        config = list(cpt["config"])
        config[-3] = cpt["weight"]["emb_g.weight"].shape[0]
        weight = {k: v for k, v in cpt["weight"].items() if "enc_q" not in k}
        operations, dtype = _make_operations(weight)
        model = cls(*config, device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=operations)
        model.load_state_dict(weight, strict=True)
        model.voice_version = version
        model.voice_if_f0 = if_f0
        model.voice_tgt_sr = config[-1]
        model.voice_n_spk = config[-3]
        model.dtype = dtype
        return (_wrap(model),)


class RVCLoaderIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rvc_name": (folder_paths.get_filename_list("rvc"), {
                "tooltip": "Exported index vectors (.npy or .pt)."}),
        }}

    RETURN_TYPES = ("RVC_INDEX",)
    RETURN_NAMES = ("index",)
    FUNCTION = "execute"
    CATEGORY = "RVC"

    def execute(self, rvc_name):
        model_path = folder_paths.get_full_path_or_raise("rvc", rvc_name)
        return (load_index_vectors(model_path),)


class RVCConvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "voice": ("RVC_VOICE", {"tooltip": "Loaded RVC voice."}),
            "hubert": ("RVC_HUBERT", {"tooltip": "Loaded RVC Hubert encoder."}),
            "pitch": ("RVC_PITCH", {"tooltip": "Loaded RMVPE or FCPE pitch model."}),
            "audio": ("AUDIO", {"tooltip": "Source audio (resampled to 16kHz mono)."}),
            "f0_method": (["rmvpe", "fcpe", "hybrid"], {"tooltip": "Pitch estimator. rmvpe/fcpe use the pitch input; hybrid averages pitch + pitch_model_b."}),
        }, "optional": {
            "pitch_model_b": ("RVC_PITCH", {"tooltip": "Second pitch model, hybrid f0 only."}),
            "f0_up_key": ("INT", {"default": 0, "min": -12, "max": 12, "step": 1}),
            "speaker_id": ("INT", {"default": 0, "min": 0, "max": 109, "step": 1}),
            "index": ("RVC_INDEX", {"tooltip": "Optional exported index vectors."}),
            "index_rate": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            "protect": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.5, "step": 0.01}),
            "rms_mix_rate": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "resample_sr": ("INT", {"default": 0, "min": 0, "max": 96000, "step": 1000, "tooltip": "Output sample rate. 0 keeps the voice rate."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = "RVC"

    @staticmethod
    def _unwrap(m):
        return m.model if isinstance(m, comfy.model_patcher.ModelPatcher) else m

    def _f0_from(self, model, wav16, thred=0.03):
        kind = getattr(model, "pitch_kind", "rmvpe")
        load_device = comfy.model_management.get_torch_device()
        dtype = model_dtype(model)
        if kind == "rmvpe":
            mel = log_mel_spectrogram(wav16.to(load_device), model.mel_basis.to(load_device))
            with torch.inference_mode():
                hidden = model.mel2hidden(mel.to(dtype))
                f0 = model.decode(hidden.float(), thred=thred)
            return f0.float().cpu().squeeze(0)
        with torch.inference_mode():
            f0 = model(wav16.to(device=load_device, dtype=dtype), threshold=0.006)
        return f0.float().cpu().squeeze(0)

    def execute(self, voice, hubert, pitch, audio, f0_method="rmvpe", pitch_model_b=None,
                f0_up_key=0, speaker_id=0, index=None, index_rate=0.75, protect=0.33,
                rms_mix_rate=1.0, resample_sr=0):
        hubert_m = self._unwrap(hubert)
        voice_m = self._unwrap(voice)
        pitch_m = self._unwrap(pitch)
        pitch_bm = self._unwrap(pitch_model_b) if pitch_model_b is not None else None
        to_load = [m for m in (voice, hubert, pitch, pitch_model_b) if m is not None]
        if any(isinstance(m, comfy.model_patcher.ModelPatcher) for m in to_load):
            comfy.model_management.load_models_gpu(to_load)
        load_device = comfy.model_management.get_torch_device()
        version = getattr(voice_m, "voice_version", "v2")
        if_f0 = int(getattr(voice_m, "voice_if_f0", 1))
        tgt_sr = int(getattr(voice_m, "voice_tgt_sr", 40000))
        n_spk = int(getattr(voice_m, "voice_n_spk", 1))
        sid = max(0, min(int(speaker_id), n_spk - 1))
        if f0_method == "hybrid" and pitch_bm is None:
            raise RuntimeError("ERROR: hybrid f0 needs pitch_model_b connected.")

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        if waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sr != 16000:
            waveform = comfy.audio.resample(waveform, sr, 16000)
        monos = [waveform[b].reshape(-1).to(torch.float32).cpu() for b in range(waveform.shape[0])]

        cuda = load_device.type == "cuda"
        pipe = RVCPipeline(tgt_sr, **({"x_pad": 3, "x_query": 10, "x_center": 60, "x_max": 65} if cuda else {"x_pad": 1, "x_query": 6, "x_center": 38, "x_max": 41}))
        outs = []
        try:
            for wav in monos:
                comfy.model_management.throw_exception_if_processing_interrupted()
                f0_hz = None
                if if_f0 == 1:
                    pad = pipe.t_pad
                    padded = torch.nn.functional.pad(wav.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)
                    kind_a = getattr(pitch_m, "pitch_kind", "rmvpe")
                    if f0_method == "hybrid":
                        fa = self._f0_from(pitch_m, padded.unsqueeze(0))
                        fb = self._f0_from(pitch_bm, padded.unsqueeze(0))
                        both = (fa > 0) & (fb > 0)
                        geo = torch.zeros_like(fa)
                        geo[both] = torch.exp(0.5 * torch.log(fa[both]) + 0.5 * torch.log(fb[both]))
                        geo[~both] = torch.where(fa[~both] > 0, fa[~both], fb[~both])
                        f0_hz = geo
                    elif f0_method != kind_a:
                        raise RuntimeError("ERROR: f0_method=%s but the pitch model is %s." % (f0_method, kind_a))
                    else:
                        f0_hz = self._f0_from(pitch_m, padded.unsqueeze(0))
                out, out_sr = pipe.convert(hubert_m, voice_m, sid, wav, f0_hz, int(f0_up_key), if_f0,
                                           index, float(index_rate), version, float(protect),
                                           float(rms_mix_rate), int(resample_sr) or tgt_sr,
                                           lambda w, a, b: comfy.audio.resample(w[None, None, :], a, b)[0, 0],
                                           load_device)
                outs.append(out.unsqueeze(0))
        finally:
            comfy.model_management.soft_empty_cache()
        wave = torch.stack(outs) if len(outs) > 1 else outs[0].unsqueeze(0)
        return ({"waveform": wave.unsqueeze(1) if wave.dim() == 2 else wave, "sample_rate": out_sr},)


NODE_CLASS_MAPPINGS = {
    "RVCLoaderHubert": RVCLoaderHubert,
    "RVCLoaderPitch": RVCLoaderPitch,
    "RVCLoaderVoice": RVCLoaderVoice,
    "RVCLoaderIndex": RVCLoaderIndex,
    "RVCConvert": RVCConvert,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RVCLoaderHubert": "Load RVC Hubert",
    "RVCLoaderPitch": "Load RVC Pitch (RMVPE/FCPE)",
    "RVCLoaderVoice": "Load RVC Voice",
    "RVCLoaderIndex": "Load RVC Index",
    "RVCConvert": "RVC Convert",
}
