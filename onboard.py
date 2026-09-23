"""Onboarding / preflight for ComfyUI-Pixal3D (envless fork).

Answers three questions before the first run on a (cloud) machine:
  1. Is this environment usable? (python / torch / CUDA / GPU / disk)
  2. Which model weights are already on the volume — including ones left
     behind by OTHER Pixal3D integrations (Saganaki22, RH-RunningHub,
     dreamrec / HF-cache snapshots) — and can they be reused instead of
     re-downloading ~26 GB?
  3. What will still be downloaded on first run?

Usage (any python 3.9+, stdlib only; torch optional for the GPU report):
    python onboard.py                  # scan + report, changes nothing
    python onboard.py --migrate        # symlink found weights into place
    python onboard.py --models-dir /path/to/ComfyUI/models   # override

Expected layout this pack uses (all under ComfyUI/models/):
    pixal3d/pipeline.json + pixal3d/ckpts/*.safetensors|*.json   (~24 GB)
    dinov3/model.safetensors                                     (~1.2 GB)
    moge/moge-2-vitl/model.pt                                    (~1.3 GB)
    naf/naf_release.pth                                          (~0.1 GB)
  optional, multi-view nodes only (downloaded on first multi-view run):
    pixal3d/pipeline_mv.json + pixal3d/ckpts/*_mv.safetensors|*.json   (~22 GB)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent

GB = 1024 ** 3

# target-relative-path -> (min plausible bytes, approx download bytes)
REQUIRED = {
    "pixal3d/pipeline.json": (100, 10_000),
    "pixal3d/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors": (int(5.0 * GB), int(5.36 * GB)),
    "pixal3d/ckpts/ss_dec_conv3d_16l8_fp16.safetensors": (int(0.12 * GB), int(0.15 * GB)),
    "pixal3d/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors": (int(5.2 * GB), int(5.55 * GB)),
    "pixal3d/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors": (int(5.2 * GB), int(5.55 * GB)),
    "pixal3d/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors": (int(5.2 * GB), int(5.55 * GB)),
    "pixal3d/ckpts/shape_dec_next_dc_f16c32_fp16.safetensors": (int(0.85 * GB), int(0.95 * GB)),
    "pixal3d/ckpts/tex_dec_next_dc_f16c32_fp16.safetensors": (int(0.85 * GB), int(0.95 * GB)),
    "pixal3d/ckpts/ss_flow_img_dit_1_3B_64_bf16.json": (10, 5_000),
    "pixal3d/ckpts/ss_dec_conv3d_16l8_fp16.json": (10, 5_000),
    "pixal3d/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json": (10, 5_000),
    "pixal3d/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json": (10, 5_000),
    "pixal3d/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json": (10, 5_000),
    "pixal3d/ckpts/shape_dec_next_dc_f16c32_fp16.json": (10, 5_000),
    "pixal3d/ckpts/tex_dec_next_dc_f16c32_fp16.json": (10, 5_000),
    "dinov3/model.safetensors": (int(0.9 * GB), int(1.21 * GB)),
    "moge/moge-2-vitl/model.pt": (int(0.8 * GB), int(1.3 * GB)),
    "naf/naf_release.pth": (int(0.05 * GB), int(0.11 * GB)),
}

# Multi-view DiTs (upstream f7cf384); decoders are shared with the entries above.
OPTIONAL_MV = {
    "pixal3d/pipeline_mv.json": (100, 10_000),
    "pixal3d/ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.safetensors": (int(5.0 * GB), int(5.36 * GB)),
    "pixal3d/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv.safetensors": (int(5.2 * GB), int(5.55 * GB)),
    "pixal3d/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv.safetensors": (int(5.2 * GB), int(5.55 * GB)),
    "pixal3d/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.safetensors": (int(5.2 * GB), int(5.55 * GB)),
    "pixal3d/ckpts/ss_flow_img_dit_1_3B_64_bf16_mv.json": (10, 5_000),
    "pixal3d/ckpts/slat_flow_img2shape_dit_1_3B_512_bf16_mv.json": (10, 5_000),
    "pixal3d/ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16_mv.json": (10, 5_000),
    "pixal3d/ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv.json": (10, 5_000),
}
ALL_WEIGHTS = {**REQUIRED, **OPTIONAL_MV}


def log(msg=""):
    print(msg, flush=True)


def human(nbytes):
    if nbytes >= GB:
        return f"{nbytes / GB:.2f} GB"
    return f"{nbytes / 1024 ** 2:.1f} MB"


# ============================================================================
# Environment report
# ============================================================================

def env_report(models_dir: Path):
    log("=== Environment ===")
    log(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    try:
        import torch
        log(f"torch       : {torch.__version__}  (CUDA {torch.version.cuda})")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            cap = torch.cuda.get_device_capability(0)
            log(f"gpu         : {p.name}  SM {cap[0]}.{cap[1]}  {p.total_memory / GB:.1f} GB VRAM")
            if cap < (8, 0):
                log("  !! SM < 8.0 — Pixal3D's flash-attn stack will not run on this GPU")
            if p.total_memory / GB >= 30:
                log("  -> vram_mode=auto will pick full_gpu (all models resident)")
            else:
                log("  -> vram_mode=auto will pick low_vram (per-stage swap)")
        else:
            log("gpu         : no CUDA device visible")
    except ImportError:
        log("torch       : not importable (run with ComfyUI's python for the GPU report)")
    log(f"git         : {'yes' if shutil.which('git') else 'MISSING (needed for MoGe/utils3d vendor install)'}")
    try:
        du = shutil.disk_usage(models_dir if models_dir.exists() else models_dir.parent)
        log(f"disk free   : {du.free / GB:.1f} GB at {models_dir}")
    except OSError:
        pass
    log()


# ============================================================================
# Candidate scan — index known layouts from other Pixal3D integrations
# ============================================================================

def _hf_hub_dirs():
    """HF cache hub dirs to scan (HF_HOME + default)."""
    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [r for r in roots if r.is_dir()]


def scan_candidates(models_dir: Path):
    """One walk over plausible roots; returns basename -> [Path, ...]."""
    wanted = {Path(rel).name for rel in ALL_WEIGHTS}
    roots = [models_dir]
    roots += _hf_hub_dirs()
    torch_hub = Path.home() / ".cache" / "torch"
    if torch_hub.is_dir():
        roots.append(torch_hub)

    index = {}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # don't descend into our vendor dir or git internals
            dirnames[:] = [d for d in dirnames if d not in (".git", "vendor", "__pycache__")]
            for fn in filenames:
                if fn in wanted:
                    index.setdefault(fn, []).append(Path(dirpath) / fn)
    return index


def _plausible(rel: str, path: Path) -> bool:
    """Reject basename collisions: generic names need a path hint + sane size."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    min_size, _ = ALL_WEIGHTS[rel]
    if size < min_size:
        return False
    s = str(path).lower()
    name = Path(rel).name
    if name == "model.safetensors":          # dinov3 backbone
        return "dinov3" in s
    if name == "model.pt":                   # MoGe-2 vitl (not the -normal variant)
        return "moge-2-vitl" in s and "normal" not in s
    return True                              # pixal3d ckpts / naf: unique basenames


def find_candidate(rel: str, index: dict):
    for path in index.get(Path(rel).name, []):
        if _plausible(rel, path):
            return path
    return None


# ============================================================================
# Report + migrate
# ============================================================================

def run(models_dir: Path, migrate: bool):
    env_report(models_dir)

    log("=== Weights inventory ===")
    index = scan_candidates(models_dir)

    missing_dl = missing_mv_dl = 0
    linked = present = suspect = 0
    to_download = []

    for rel, (min_size, approx) in ALL_WEIGHTS.items():
        optional = rel in OPTIONAL_MV
        target = models_dir / rel
        if target.exists():
            size = target.stat().st_size
            if size >= min_size:
                present += 1
                continue
            log(f"SUSPECT  {rel}  ({human(size)}, expected >= {human(min_size)})")
            log(f"         looks like a partial download — delete it and re-run, or let the node re-fetch")
            suspect += 1
            continue

        cand = find_candidate(rel, index)
        if cand is not None and cand.resolve() != target.resolve():
            if migrate:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.symlink(cand, target)
                    log(f"LINKED   {rel}  <-  {cand}")
                except OSError as e:
                    log(f"COPYING  {rel}  <-  {cand}  (symlink failed: {e})")
                    shutil.copy2(cand, target)
                linked += 1
            else:
                log(f"FOUND    {rel}  at  {cand}  (run with --migrate to link)")
                linked += 1
        elif optional:
            log(f"MISSING  {rel}  (multi-view only; ~{human(approx)} download on first multi-view run)")
            missing_mv_dl += approx
        else:
            log(f"MISSING  {rel}  (~{human(approx)} download on first run)")
            to_download.append(rel)
            missing_dl += approx

    log()
    log("=== Summary ===")
    log(f"present: {present}   reusable elsewhere: {linked}   suspect: {suspect}   missing: {len(to_download)}")
    if to_download:
        log(f"first run will download ~{human(missing_dl)}")
    else:
        log("no downloads needed on first run")
    if missing_mv_dl:
        log(f"multi-view nodes will download ~{human(missing_mv_dl)} more on their first run")
    if not migrate and linked:
        log("re-run with --migrate to symlink the reusable files into place")
    log()
    log("next steps:")
    log("  1. python install.py          (build vendor/ for this machine)")
    log("  2. python install.py --check  (every line should say OK)")
    log("  3. restart ComfyUI, load workflows/pixal3d_basic.json")


def main():
    ap = argparse.ArgumentParser(description="ComfyUI-Pixal3D onboarding preflight")
    ap.add_argument("--migrate", action="store_true",
                    help="symlink weights found in other layouts into this pack's layout")
    ap.add_argument("--models-dir", type=Path, default=None,
                    help="ComfyUI models dir (default: ../../models relative to this pack)")
    args = ap.parse_args()

    models_dir = args.models_dir or PACK_DIR.parents[1] / "models"
    if not models_dir.is_dir():
        log(f"models dir not found at {models_dir} — pass --models-dir /path/to/ComfyUI/models")
        sys.exit(1)

    run(models_dir.resolve(), migrate=args.migrate)


if __name__ == "__main__":
    main()
