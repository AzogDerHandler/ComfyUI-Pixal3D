# Model inventory — where every weight lives

Verified against the RunComfy volume on 2026-07-27 (`ls -lR models/Pixal3D`,
ComfyUI at `/workspace/ComfyUI`). All sizes matched the expected byte counts
exactly — no truncated downloads.

## Source of truth on the volume (pre-existing downloads)

| Model | Path under `ComfyUI/models/` | Size | Status |
|---|---|---|---|
| Pixal3D cascade (7 ckpts + 7 json + pipeline.json) | `Pixal3D/TencentARC_Pixal3D/` | 24.0 GB | ✅ complete, verified |
| ~~Duplicate of the above~~ | `Pixal3D/ckpts/` + `Pixal3D/pipeline.json` | 24.0 GB | ⚠️ byte-identical duplicate — safe to delete |
| MoGe-2 ViT-L (camera estimation) | `Pixal3D/MoGe/moge-2-vitl/model.pt` | 1.32 GB | ✅ verified |
| DINOv3 ViT-L backbone | `Pixal3D/camenduru_dinov3-vitl16-pretrain-lvd1689m/model.safetensors` | 1.21 GB | ✅ verified |
| NAF upsampler | `Pixal3D/torch_hub/` | empty | ❌ missing — auto-downloads ~100 MB to `models/naf/` on first run |
| RMBG-2.0 (BiRefNet + ONNX variants) | `Pixal3D/RMBG-2.0/` | ~10.4 GB | 🗑️ NOT used by this pack (rembg is stubbed; wire a MASK instead) — deletable |

## Where this pack looks for weights (post-`--migrate` layout)

`install.py` / `onboard.py --migrate` symlink the sources above into:

| Loader expects | Symlinked from |
|---|---|
| `models/pixal3d/pipeline.json` + `models/pixal3d/ckpts/*` | `models/Pixal3D/TencentARC_Pixal3D/…` (or the duplicate, whichever the scan hits) |
| `models/dinov3/model.safetensors` | `models/Pixal3D/camenduru_dinov3…/model.safetensors` |
| `models/moge/moge-2-vitl/model.pt` | `models/Pixal3D/MoGe/moge-2-vitl/model.pt` |
| `models/naf/naf_release.pth` | (downloaded from the valeoai/NAF GitHub release on first run) |

Note: `models/pixal3d` (lowercase, this pack) and `models/Pixal3D` (uppercase,
legacy downloads) are different directories on Linux — the lowercase one holds
only symlinks into the uppercase one.

## Housekeeping

- **Delete duplicates BEFORE running install/onboard** (or re-run onboarding
  after deleting): the migrate step links against whichever copy it finds, and
  deleting the linked copy afterwards leaves dangling symlinks.
  ```bash
  rm -r models/Pixal3D/ckpts models/Pixal3D/pipeline.json   # frees ~24 GB
  rm -r models/Pixal3D/RMBG-2.0                             # frees ~10 GB (only if no other pack uses it)
  ```
- Directory permission bits on the volume are odd (`drw-r--r--`, no execute
  bit) — harmless while ComfyUI runs as root (it does on RunComfy), but if a
  future setup runs as a normal user, unreadable-directory errors trace back
  to this.
- The volume also stores `_pixal3d_comfy_pipeline.json` (Saganaki wrapper
  artifact) and HF snapshot metadata (`LICENSE`, `NOTICE`, `README.md`,
  `.gitattributes`) — all ignored by this pack.
