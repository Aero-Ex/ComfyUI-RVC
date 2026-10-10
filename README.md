# ComfyUI-RVC

Native, dependency-free Retrieval-based Voice Conversion (RVC) for ComfyUI.
Inference only — no training, no index building.

## Models

Download the base assets from [Aero-Ex/ComfyUI-RVC](https://huggingface.co/Aero-Ex/ComfyUI-RVC)
on Hugging Face and place them in `ComfyUI/models/rvc/`. The three base assets
are Comfy-format safetensors; voices are used directly from upstream `.pth`.

| File | What it is |
| --- | --- |
| `hubert_base.safetensors` | ContentVec-500 content encoder |
| `rmvpe.safetensors` | RMVPE pitch estimator |
| `fcpe.safetensors` | FCPE pitch estimator (fast) |
| `<voice>.pth` | An RVC voice model |

`hubert_base.safetensors`, `rmvpe.safetensors` and `fcpe.safetensors` are
converted from the upstream `.pt` files with the pack's offline converter.
Voice `.pth` files are RVC checkpoints with the keys `weight`, `config`,
`info`, `sr`, `f0`, `version` — put them in the same folder, no conversion
needed.

The RVC voice file is the one thing you must supply yourself. Any RVC v1 or
v2 model from a voice-model sharing site works. Do **not** use a raw
training checkpoint (`G_*.pth` / `D_*.pth` with only `model`) — the loader
requires the RVC wrapper format above and will reject it.

## Workflow

```
Load RVC Hubert ─────────┐
Load RVC Pitch (RMVPE) ──┼─→ RVC Convert ─→ Save Audio
Load RVC Voice ──────────┘        ↑
                              Load Audio
```

1. **Load RVC Hubert** — pick `hubert_base.safetensors`.
2. **Load RVC Pitch (RMVPE/FCPE)** — pick `rmvpe.safetensors` or `fcpe.safetensors`.
3. **Load RVC Voice** — pick your voice `.pth`.
4. **Load Audio** — any source audio. Stereo and arbitrary sample rates are
   handled for you (resampled to 16 kHz mono internally).
5. **RVC Convert** — set `f0_method` to match the pitch model you loaded.
6. Feed the output into `Preview Audio` or `Save Audio`.

## Node inputs

### Loaders

`Load RVC Hubert`, `Load RVC Pitch`,  each
take a filename from `models/rvc/`.

### RVC Convert

| Input | Default | Notes |
| --- | --- | --- |
| `f0_method` | `rmvpe` | Must match the loaded pitch model, except `hybrid` (below). |
| `f0_up_key` | 0 | Pitch shift in semitones, −12…12. |
| `speaker_id` | 0 | Speaker slot for multi-speaker voices. Range auto-clamps to the voice. |
| `index_rate` | 0.75 | Retrieval blend strength. Only used when an index is connected. |
| `protect` | 0.33 | Protects unvoiced/consonant sounds from the f0 path. 0.33 is the upstream default; 0.5 disables the blend entirely. |
| `rms_mix_rate` | 1.0 | Volume-envelope matching against the input. 1.0 = off. |
| `resample_sr` | 0 | Output sample rate. 0 keeps the voice's own rate. |

`pitch_model_b` is only needed for `f0_method="hybrid"`, which averages the
primary and secondary pitch estimators. Leave it unconnected otherwise —
connecting only `pitch_model_a` with `hybrid` selected raises a clear error.

## Notes

- **Output is stochastic.** RVC samples from a posterior at inference, so two
  runs on identical input differ by roughly 0.4–0.7 in waveform amplitude.
  This matches upstream, which seeds once at GUI startup rather than per
  conversion. It is not a bug and not fixable without changing the model's
  math.
- **VRAM** is dynamic. All three base models plus a voice run comfortably on
  a 6 GB card at fp16.
- **Sample rate** of the output is the voice model's native rate unless
  `resample_sr` is set. Upstream uses `librosa.resample`; this pack uses
  `comfy.audio.resample` so it matches the rest of ComfyUI's audio handling.
# ComfyUI-RVC
