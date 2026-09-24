# ComfyUI-Pixal3D (envless fork)

ComfyUI nodes for **Pixal3D** (SIGGRAPH 2026, TencentARC) — pixel-aligned image-to-3D
generation. Single image — or several posed views — in, textured PBR GLB out.

Fork of [PozzettiAndrea/ComfyUI-Pixal3D](https://github.com/PozzettiAndrea/ComfyUI-Pixal3D),
rebuilt for **cloud GPU providers (RunComfy, RunPod, ...)** and **max quality on
big-VRAM machines**:

- **No comfy-env / pixi.** Every dependency — including the compiled CUDA stack
  (flash-attn, natten, flex_gemm, cumesh, o_voxel, drtk) — installs into
  `vendor/` **inside this folder** via prebuilt wheels. Since `custom_nodes`
  lives on the persisted volume, dependencies survive pod restarts. No source
  builds, no CUDA toolkit, no host-env writes.
- **`vram_mode=full_gpu`**: all 13 models stay resident on the GPU. No
  per-stage CPU↔GPU swapping (the upstream `low_vram` behavior that this pack
  previously hard-coded). `auto` picks full_gpu on cards with ≥ 30 GB VRAM.
- **Self-contained camera estimation**: new `Pixal3DEstimateCamera` node runs
  the vendored MoGe-2 — the companion ComfyUI-MoGe2 pack is no longer needed.
- **Max-quality defaults**: `1536_cascade` default, `max_num_tokens` up to
  262144, 8K texture bake on the split PBR chain, up to 5M faces.
- FlexGEMM's triton autotune cache is persisted next to the pack (kills the
  "first run after every restart is slow" tax on ephemeral pods).
- **Multi-view** (upstream's Sep 2026 release): condition the cascade on 2–9+
  views of the same object — orbit-video frames, multi-view generator output,
  or a dataset folder — with a built-in rig check before the run.

Upstream model: [code](https://github.com/TencentARC/Pixal3D) ·
[weights](https://huggingface.co/TencentARC/Pixal3D) ·
[paper](https://arxiv.org/abs/2605.10922). Code and weights are **MIT**.

## Install (cloud pod / Linux)

**Option A — ComfyUI-Manager (works on restricted cloud shells).** Some
providers (e.g. RunComfy's web terminal) whitelist shell commands and block
`python`. Use Manager's **"Install via Git URL"** with
`https://github.com/AzogDerSchaender/ComfyUI-Pixal3D.git` — Manager runs
`install.py` with ComfyUI's own python, which builds `vendor/`, verifies every
import, AND auto-runs the onboarding weight migration. Watch the server log
for `[pixal3d-install]` lines, then restart ComfyUI.

**Option B — shell (when `python` is available):**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AzogDerSchaender/ComfyUI-Pixal3D.git
cd ComfyUI-Pixal3D
python onboard.py        # preflight: env report + weight inventory
python install.py        # builds vendor/ + auto-runs onboarding --migrate
```

`onboard.py` scans `ComfyUI/models/` and the HF/torch caches for Pixal3D,
DINOv3, MoGe-2, and NAF weights downloaded by any previous integration
(Saganaki22, RH-RunningHub, dreamrec/HF snapshots) and symlinks them into
this pack's layout — so you don't re-download ~26 GB you already have. It
also flags partial/corrupt downloads (delete those; they'd otherwise be
skipped by the downloader and crash at load).

`install.py` detects the host Python/torch/CUDA ABI, resolves matching
prebuilt wheels from the [cuda-wheels index](https://github.com/PozzettiAndrea/cuda-wheels),
installs everything into `vendor/`, and verifies every import. Then restart
ComfyUI. Sanity check any time with:

```bash
python install.py --check
```

Model weights (~23 GB Pixal3D + 1.2 GB DINOv3 + 1.3 GB MoGe-2 + 3 MB NAF)
auto-download to `ComfyUI/models/` on first run — also on the persisted volume.
The multi-view nodes add ~22 GB (`ckpts/*_mv`) on their first run.

Useful env vars: `PIXAL3D_WHEEL_INDEX` (alternate wheel index),
`PIXAL3D_VENDOR_APPEND=1` (host packages win over vendor pins),
`PIXAL3D_VENDOR_DISABLE=1` (ignore vendor entirely),
`PIXAL3D_SKIP_INSTALL=1` (make install.py a no-op).

If you switch to a machine with a different Python/torch/CUDA, re-run
`python install.py` once (a startup log warning will tell you).

## Nodes

| Node | Purpose |
|------|---------|
| `Pixal3DLoadPipeline` | Thin config: `pipeline_type` (1024/1536 cascade), `attn_backend`, `vram_mode` (auto / full_gpu / low_vram). |
| `Pixal3DPreprocessImage` | Pure-PIL alpha-bbox crop + resize (bring your own MASK for bg removal). |
| `Pixal3DEstimateCamera` | MoGe-2 FOV estimation → camera dict. Feed the ORIGINAL (pre-crop) image. |
| `Pixal3DCameraFromFOV` | Manual FOV → camera dict (bypass MoGe for synthetic/known cameras). |
| `Pixal3DGenerateGLB` | Fused cascade → vertex-color GLB (fast convenience path, no UV textures). |
| `Pixal3DGenerateMesh` → `Pixal3DProcessMesh` → `Pixal3DRasterizePBR` → `Pixal3DExportGLB` | **The quality path**: raw mesh + PBR voxel grid → cleanup/UV unwrap → UV-space PBR bake (up to 8192px) → GLB. |
| `Pixal3DMultiViewInput` | IMAGE batch of views (+ MASK batch) + orbit rig (azimuths / elevations / FOV) → multi-view bundle, preview, report. |
| `Pixal3DLoadMultiViewFolder` | Upstream-format folder (`transforms.json` + RGBA views) → multi-view bundle. Empty path = bundled upstream example. |
| `Pixal3DGenerateMeshMV` / `Pixal3DGenerateGLBMV` | Multi-view twins of GenerateMesh / GenerateGLB. The mesh output plugs into the same ProcessMesh → RasterizePBR → ExportGLB chain. |

## Multi-view

Upstream's multi-view release (TencentARC/Pixal3D `f7cf384`) conditions the
same four-stage cascade on V views at once: every view's DINOv3+NAF features
are projected into the shared voxel grid through its camera and averaged. It
uses separate `ckpts/*_mv` flow DiTs (~22 GB, downloaded on the first
multi-view run); decoders, DINOv3 and NAF are shared with single-view (one copy
in memory). Start from `workflows/pixal3d_multiview.json`, which runs the
bundled upstream example end to end.

**The cameras must be known and the views must agree with them:**

- **View 0 is the front view.** The mesh is posed in its camera frame, so an
  eye-level front view gives an upright mesh (`front_index` picks it).
- **Azimuth direction:** +90° means the camera moved toward the side that is on
  the **right of the front image** (upstream: camera at +X, sees the object's
  left). If your frames rotate the other way set `view_at_plus_90 =
  front_image_left_side` — the report tells you when the other direction fits
  the silhouettes better.
- **One shared scale and orbit center.** `align_views=bbox_height` (default)
  rescales every view so the object's bbox height matches and recenters it —
  on an eye-level orbit the height is the same from every side, so this repairs
  frames of different sizes or crops. Use `align_views=none` for frames that
  already share one camera (renders, one uncropped video) — perspective makes
  heights differ by a few percent, so as-given is more exact there — and for
  elevated orbits, where the height legitimately changes per view. With `none`,
  non-square frames are only padded (FOV corrected); upstream would squash them.
- **FOV** is the horizontal FOV of the front frame as given. For video frames,
  run `Pixal3DEstimateCamera` on the front frame and wire `fov_x_deg`.
- **Masks:** no background removal happens here — feed RGBA, or a MASK batch
  from any rem-bg node (`invert_mask` for LoadImage's inverted MASK).

`framing=auto_fit` (default) derives the camera distance from the silhouettes
so the object fills `1/grid_fill_margin` of the voxel grid — the same framing
single-view uses. `upstream_rig` reproduces upstream's example rig (unit cube
at 1/1.1 of the frame, distance `1.1·0.5/tan(fov/2)`).

**Rig check.** Before the cascade, the input node carves a visual hull from all
silhouettes through the rig and reprojects it into every view. The `preview`
shows what the rig can't explain in red (plus the voxel grid's outline at the
object's depth in yellow — the object must fit inside — with its +X side, the
object's left, in green); the `report` gives per-view
silhouette coverage (a consistent rig scores ~95–100%) and flags a flipped
rotation direction, off-center or clipped objects, and a non-front view 0. It
is cheap — check it before spending minutes on a run.

## Max-quality recipe (big cloud GPU)

- `Pixal3DLoadPipeline`: `1536_cascade`, `vram_mode=full_gpu`, `attn_backend=auto`
- `Pixal3DGenerateMesh`: `max_num_tokens` 98304–131072, steps 16/16/16
- `Pixal3DProcessMesh`: `target_face_count` to taste (up to 5M)
- `Pixal3DRasterizePBR`: `texture_size` 4096–8192, wire `original_mesh` from
  GenerateMesh for BVH-snapped (sharper) textures
- Use `workflows/pixal3d_basic.json` as the starting graph.

## Conflicts

visualbruno's **ComfyUI-Trellis2** loads its own builds of the same compiled
libraries (`cumesh`, `o_voxel`) at startup; pybind11 refuses a second copy in one
process (`generic_type: type "CuMesh" is already registered!`). The two packs
can't run in the same ComfyUI — disable one (rename its `custom_nodes` folder to
end in `.disabled`) and restart.

## Hardware

- NVIDIA GPU, **SM ≥ 8.0** (Ampere/Ada/Hopper/Blackwell).
- `full_gpu` + 1536_cascade wants ~40 GB+ VRAM; `low_vram` + 1024_cascade fits 24 GB.
- ~30 GB disk for weights (+22 GB with multi-view), ~5 GB for vendor/.
- Multi-view in `full_gpu`: only one variant's flow DiTs (~11 GB) is resident at
  a time; switching between single- and multi-view swaps them.

## Credits

Pixal3D authors (Tsinghua / Tencent ARC / VUW); Microsoft TRELLIS.2 & MoGe;
Meta DINOv3; original ComfyUI wrapper and cuda-wheels index by Andrea Pozzetti.
