import folder_paths
import comfy.audio
import comfy.utils
import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.storage
import torch

from .rvc.hubert import HubertEncoder
from .rvc.rmvpe import RMVPEModel
from .rvc.fcpe import FCPitch
from .rvc.index import load_index_vectors
from .rvc.pipeline import RVCPipeline
from .rvc.synth import SAMPLE_RATES, Synthesizer


def _make_operations(sd):
    device = comfy.model_management.get_torch_device()
    dtype = comfy.model_management.unet_dtype(model_params=-1, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32])
    manual_cast_dtype = comfy.model_management.unet_manual_cast(dtype, device, supported_dtypes=[torch.float16, torch.bfloat16, torch.float32])
    quant = comfy.utils.detect_layer_quantization(sd, "")
    if quant is not None:
        return comfy.ops.mixed_precision_ops(quant, dtype), dtype
    return comfy.ops.pick_operations(dtype, manual_cast_dtype, load_device=device), dtype


def _load(cls, sd, *args, fast_disk=False):
    operations, dtype = _make_operations(sd)
    model = cls(*args, device=comfy.model_management.unet_offload_device(), dtype=dtype, operations=operations)
    model.load_state_dict(sd, strict=True)
    model.eval()  # RMVPE's BatchNorm must use its running statistics
    return comfy.model_patcher.CoreModelPatcher(
        model, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device(), fast_disk=fast_disk)


def _load_safetensors(rvc_name):
    model_path = folder_paths.get_full_path_or_raise("rvc", rvc_name)
    sd, metadata = comfy.utils.load_torch_file(model_path, safe_load=True, return_metadata=True)
    return sd, metadata or {}, comfy.storage.state_dict_fast_disk(sd)


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
        sd, metadata, fast_disk = _load_safetensors(rvc_name)
        if metadata.get("rvc_hubert.format_version") != "1" or "post_extract_proj.weight" not in sd:
            raise RuntimeError("ERROR: not a converted RVC Hubert checkpoint: %s" % rvc_name)
        return (_load(HubertEncoder, sd, fast_disk=fast_disk),)


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
        sd, metadata, fast_disk = _load_safetensors(rvc_name)
        if metadata.get("rvc_rmvpe.format_version") == "1" and "fc.1.weight" in sd:
            return (_load(RMVPEModel, sd, fast_disk=fast_disk),)
        if metadata.get("rvc_fcpe.format_version") == "1" and "input_stack.0.weight" in sd:
            return (_load(FCPitch, sd, fast_disk=fast_disk),)
        raise RuntimeError("ERROR: not a converted RVC pitch checkpoint: %s" % rvc_name)


class RVCVoice:
    """A loaded RVC voice: the synthesizer patcher plus the checkpoint facts the converter needs."""

    def __init__(self, patcher, version, has_f0, tgt_sr, n_spk):
        self.patcher = patcher
        self.version = version
        self.has_f0 = has_f0
        self.tgt_sr = tgt_sr
        self.n_spk = n_spk


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
        if version not in ("v1", "v2"):
            raise RuntimeError("ERROR: unknown RVC voice version: %s" % version)
        has_f0 = int(cpt.get("f0", 1)) == 1
        config = list(cpt["config"])
        n_spk = cpt["weight"]["emb_g.weight"].shape[0]
        config[-3] = n_spk
        weight = {k: v for k, v in cpt["weight"].items() if not k.startswith("enc_q.")}
        patcher = _load(Synthesizer, weight, config, 256 if version == "v1" else 768, has_f0)
        return (RVCVoice(patcher, version, has_f0, SAMPLE_RATES.get(config[-1], config[-1]), n_spk),)


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
        }, "optional": {
            "pitch_model_b": ("RVC_PITCH", {"tooltip": "Optional second pitch model. When connected, f0 is the mean of both where both are voiced."}),
            "f0_up_key": ("INT", {"default": 0, "min": -24, "max": 24, "step": 1, "tooltip": "Pitch shift in semitones."}),
            "speaker_id": ("INT", {"default": 0, "min": 0, "max": 1023, "step": 1, "tooltip": "Speaker slot for multi-speaker voices; clamped to the voice."}),
            "index": ("RVC_INDEX", {"tooltip": "Optional exported index vectors."}),
            "index_rate": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05}),
            "protect": ("FLOAT", {"default": 0.33, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "How much of the indexed features unvoiced frames keep. 0.5 disables protection."}),
            "rms_mix_rate": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "resample_sr": ("INT", {"default": 0, "min": 0, "max": 96000, "step": 1000, "tooltip": "Output sample rate. 0 keeps the voice rate."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Seed for the voice's latent and excitation noise."}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "execute"
    CATEGORY = "RVC"

    def execute(self, voice, hubert, pitch, audio, pitch_model_b=None, f0_up_key=0, speaker_id=0, index=None,
                index_rate=0.75, protect=0.33, rms_mix_rate=1.0, resample_sr=0, seed=0):
        pitch_models = [pitch] + ([pitch_model_b] if pitch_model_b is not None else []) if voice.has_f0 else []
        comfy.model_management.load_models_gpu([voice.patcher, hubert] + pitch_models)
        device = comfy.model_management.get_torch_device()
        generator = torch.Generator(device=device).manual_seed(seed)
        sid = min(speaker_id, voice.n_spk - 1)
        # Upstream sizes the segments by precision: half-precision runs use more context per segment.
        half = voice.patcher.model.dtype == torch.float16
        pipe = RVCPipeline(voice.tgt_sr, **({"x_pad": 3, "x_query": 10, "x_center": 60, "x_max": 65} if half else {"x_pad": 1, "x_query": 6, "x_center": 38, "x_max": 41}))

        waveform = audio["waveform"].mean(dim=1)
        if audio["sample_rate"] != 16000:
            waveform = comfy.audio.resample(waveform, audio["sample_rate"], 16000)
        outs = []
        for wav in waveform.float().cpu():
            comfy.model_management.throw_exception_if_processing_interrupted()
            out, out_sr = pipe.convert(hubert.model, voice.patcher.model, sid, wav, [m.model for m in pitch_models], f0_up_key,
                                       index, index_rate, voice.version, protect, rms_mix_rate,
                                       resample_sr or voice.tgt_sr, device, generator)
            outs.append(out)
        return ({"waveform": torch.stack(outs).unsqueeze(1), "sample_rate": out_sr},)


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
