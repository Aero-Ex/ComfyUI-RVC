# ComfyUI-RVC

dependency-free Retrieval-based Voice Conversion (RVC) for ComfyUI.
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
5. **RVC Convert** — connect the three models and the audio.
6. Feed the output into `Preview Audio` or `Save Audio`.

## Node inputs

### Loaders

`Load RVC Hubert`, `Load RVC Pitch`,  each
take a filename from `models/rvc/`.

### RVC Convert

| Input | Default | Notes |
| --- | --- | --- |
| `f0_up_key` | 0 | Pitch shift in semitones, −24…24. |
| `speaker_id` | 0 | Speaker slot for multi-speaker voices. Range auto-clamps to the voice. |
| `index_rate` | 0.75 | Retrieval blend strength. Only used when an index is connected. |
| `protect` | 0.33 | Upstream's consonant protection: frames at 0 Hz keep only this share of the index-retrieved features. Like upstream, unvoiced frames are interpolated first, so it only acts when no frame is voiced. 0.5 disables it. |
| `rms_mix_rate` | 1.0 | Volume-envelope matching against the input. 1.0 = off. |
| `resample_sr` | 0 | Output sample rate. 0 keeps the voice's own rate. |
| `seed` | 0 | Seed for the voice's latent and excitation noise. Same seed, same output. |

The pitch estimator is whichever model is connected to `pitch`. Connecting a
second one to `pitch_model_b` averages the two where both hear a voiced frame.

## Notes

- **Output is sampled.** RVC draws latent noise at inference; the `seed`
  input fixes it, so a seed always gives the same conversion.
- **VRAM** is managed by ComfyUI's dynamic loader. All three base models plus a voice run comfortably on
  a 6 GB card at fp16.
- **Sample rate** of the output is the voice model's native rate unless
  `resample_sr` is set. Upstream uses `librosa.resample`; this pack uses
  `comfy.audio.resample` so it matches the rest of ComfyUI's audio handling.
# ComfyUI-RVC
