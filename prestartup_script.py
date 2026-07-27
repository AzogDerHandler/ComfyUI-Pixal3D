"""Prestartup for ComfyUI-Pixal3D (envless fork).

Runs before ComfyUI imports anything heavy, which makes it the one reliable
place to (a) put vendor/ at the FRONT of sys.path so our pinned libs win over
whatever the host image ships, and (b) set env vars that must exist before
torch/cv2 initialize.

Escape hatches:
    PIXAL3D_VENDOR_DISABLE=1  don't touch sys.path (use host packages)
    PIXAL3D_VENDOR_APPEND=1   append vendor/ instead of prepending (host wins
                              on conflicts; only packages absent from the host
                              resolve from vendor/)
"""

import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / "vendor"
COMFYUI_DIR = SCRIPT_DIR.parent.parent

if VENDOR_DIR.is_dir() and os.environ.get("PIXAL3D_VENDOR_DISABLE") != "1":
    _v = str(VENDOR_DIR)
    if _v in sys.path:
        sys.path.remove(_v)
    if os.environ.get("PIXAL3D_VENDOR_APPEND") == "1":
        sys.path.append(_v)
    else:
        sys.path.insert(0, _v)
    print(f"[pixal3d] vendor on sys.path: {_v}", flush=True)

# Must be set before torch allocates; harmless if the host already set it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
# FlexGEMM's triton autotune is the "first run is slow" tax. Point its cache
# at the pack dir, which persists on cloud volumes (best-effort: flex_gemm
# builds that don't read this var just ignore it).
os.environ.setdefault(
    "FLEX_GEMM_AUTOTUNE_CACHE_PATH", str(SCRIPT_DIR / "autotune_cache.json")
)

# Copy bundled sample images into ComfyUI/input (skip ones already there).
_input_dir = COMFYUI_DIR / "input"
_assets = SCRIPT_DIR / "assets"
if _assets.is_dir() and _input_dir.is_dir():
    for f in _assets.iterdir():
        if f.is_file() and not (_input_dir / f.name).exists():
            try:
                shutil.copy2(f, _input_dir / f.name)
            except OSError:
                pass
