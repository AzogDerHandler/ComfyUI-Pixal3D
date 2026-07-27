# ComfyUI-Pixal3D (envless fork)

ComfyUI nodes for **Pixal3D** (SIGGRAPH 2026, TencentARC) — pixel-aligned image-to-3D
generation. Single image in, textured PBR GLB out.

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

Upstream model: [code](https://github.com/TencentARC/Pixal3D) ·
[weights](https://huggingface.co/TencentARC/Pixal3D) ·
[paper](https://arxiv.org/abs/2605.10922). Code and weights are **MIT**.

## Install (cloud pod / Linux)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AzogDerSchaender/ComfyUI-Pixal3D.git
cd ComfyUI-Pixal3D
python install.py        # use ComfyUI's own python (its venv python)
```

`install.py` detects the host Python/torch/CUDA ABI, resolves matching
prebuilt wheels from the [cuda-wheels index](https://github.com/PozzettiAndrea/cuda-wheels),
installs everything into `vendor/`, and verifies every import. Then restart
ComfyUI. Sanity check any time with:

```bash
python install.py --check
```

Model weights (~23 GB Pixal3D + 1.2 GB DINOv3 + 1.3 GB MoGe-2 + 100 MB NAF)
auto-download to `ComfyUI/models/` on first run — also on the persisted volume.

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

## Max-quality recipe (big cloud GPU)

- `Pixal3DLoadPipeline`: `1536_cascade`, `vram_mode=full_gpu`, `attn_backend=auto`
- `Pixal3DGenerateMesh`: `max_num_tokens` 98304–131072, steps 16/16/16
- `Pixal3DProcessMesh`: `target_face_count` to taste (up to 5M)
- `Pixal3DRasterizePBR`: `texture_size` 4096–8192, wire `original_mesh` from
  GenerateMesh for BVH-snapped (sharper) textures
- Use `workflows/pixal3d_basic.json` as the starting graph.

## Hardware

- NVIDIA GPU, **SM ≥ 8.0** (Ampere/Ada/Hopper/Blackwell).
- `full_gpu` + 1536_cascade wants ~40 GB+ VRAM; `low_vram` + 1024_cascade fits 24 GB.
- ~30 GB disk for weights, ~5 GB for vendor/.

## Credits

Pixal3D authors (Tsinghua / Tencent ARC / VUW); Microsoft TRELLIS.2 & MoGe;
Meta DINOv3; original ComfyUI wrapper and cuda-wheels index by Andrea Pozzetti.
