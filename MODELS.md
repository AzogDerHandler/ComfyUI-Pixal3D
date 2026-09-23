# Model inventory — where every weight lives

Verified on the RunComfy volume 2026-07-27 (`ls -lR`, ComfyUI at
`/workspace/ComfyUI`). Every file matched the expected byte count exactly —
no truncated downloads. The former 24 GB duplicate `models/Pixal3D/ckpts/`
copy was deleted; exactly one copy of each weight remains.

## Canonical layout (what this pack reads)

Weights were MOVED (`mv`) out of the legacy `models/Pixal3D/` download
locations into the paths the loaders use:

| Loader expects (under `ComfyUI/models/`) | Size | Origin |
|---|---|---|
| `pixal3d/pipeline.json` + `pixal3d/ckpts/` (7 safetensors + 7 json) | 24.0 GB | moved from `Pixal3D/TencentARC_Pixal3D/` (pipeline.json copied, snapshot keeps its own) |
| `dinov3/model.safetensors` | 1.21 GB | moved from `Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/` |
| `moge/moge-2-vitl/model.pt` | 1.32 GB | moved from `Pixal3D/MoGe/moge-2-vitl/` |
| `naf/naf_release.pth` | ~100 MB | auto-downloads from the valeoai/NAF GitHub release on first run |

Expected ckpt byte sizes (for future truncation checks):
ss_flow 5,359,822,584 · img2shape_512 5,546,764,048 · img2shape_1024
5,546,764,048 · imgshape2tex_1024 5,546,960,656 · shape_dec 948,490,494 ·
tex_dec 948,458,812 · ss_dec 147,591,972 · pipeline.json 4,068.

## Multi-view weights (optional)

Only the multi-view nodes use these; they download on the first multi-view run
into the same `pixal3d/` folder (not yet on the volume as of 2026-09-23):

| File (under `ComfyUI/models/pixal3d/`) | Bytes |
|---|---|
| `pipeline_mv.json` | ~4 KB |
| `ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.safetensors` | 5,359,822,584 |
| `ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv.safetensors` | 5,546,764,048 |
| `ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv.safetensors` | 5,546,764,048 |
| `ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.safetensors` | 5,546,960,656 |
| `ckpts/*_mv.json` (4) | ~0.5 KB each |

~22 GB total. The decoders are shared with single-view. Deleting the unused
`models/Pixal3D/RMBG-2.0/` (~10.4 GB, below) frees half of that.

## Legacy leftovers under `models/Pixal3D/` (ignored by this pack)

| Path | Size | Note |
|---|---|---|
| `RMBG-2.0/` | ~10.4 GB | NOT used (rembg is stubbed — wire a MASK instead). Kept by choice; deletable any time, incl. 3.5 GB of ONNX variants nothing loads. |
| `TencentARC_Pixal3D/` (LICENSE, NOTICE, README, pipeline.json, `_pixal3d_comfy_pipeline.json`) | ~40 KB | HF snapshot metadata + a Saganaki wrapper artifact; ckpts were moved out. |
| `camenduru_dinov3…/` (config.json, preprocessor_config.json, READMEs) | ~25 KB | Snapshot metadata; model.safetensors was moved out. |
| `torch_hub/`, `MoGe/` | empty | leftover dirs. |

## Notes

- `models/pixal3d` (lowercase, this pack) vs `models/Pixal3D` (uppercase,
  legacy) are distinct directories on Linux.
- Directory permission bits on the volume are odd (`drw-r--r--`, no execute
  bit) — harmless while ComfyUI runs as root (RunComfy does), but a non-root
  setup would hit unreadable-directory errors here.
- RunComfy's web terminal whitelists commands (no `python`, `find`, `ln`) —
  which is why migration used `mv` instead of onboard.py's symlinks, and why
  `vendor/` is built with hand-issued `pip install --target` commands or via
  ComfyUI-Manager running install.py.
