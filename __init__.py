"""ComfyUI-Pixal3D (envless fork) — node registration.

All heavy dependencies live in <pack>/vendor/ (see install.py). The vendor
path is normally prepended to sys.path by prestartup_script.py; the fallback
insert below covers embeds that skip prestartup scripts. Node modules only
import torch/comfy_api at module level — the compiled CUDA stack loads lazily
at first node execution, so ComfyUI always boots even with an empty vendor/.
"""

import json
import logging
import os
import sys
import traceback
from pathlib import Path

log = logging.getLogger("pixal3d")

_HERE = Path(__file__).resolve().parent
_VENDOR = _HERE / "vendor"

if (
    _VENDOR.is_dir()
    and str(_VENDOR) not in sys.path
    and os.environ.get("PIXAL3D_VENDOR_DISABLE") != "1"
):
    sys.path.insert(0, str(_VENDOR))

# Warn (don't fail) if vendor/ was built against a different ABI than the
# torch we're running on — the classic symptom is an ImportError deep inside
# flash_attn/flex_gemm, and this message names the actual cause.
_manifest = _VENDOR / ".pixal3d_vendor.json"
if _manifest.is_file():
    try:
        import torch

        _m = json.loads(_manifest.read_text())
        _torch_now = torch.__version__.split("+")[0]
        _cuda_now = "cu" + (torch.version.cuda or "0.0").replace(".", "")
        _py_now = f"cp{sys.version_info.major}{sys.version_info.minor}"
        if (_m.get("torch"), _m.get("cuda"), _m.get("python")) != (
            _torch_now, _cuda_now, _py_now,
        ):
            log.warning(
                "[ComfyUI-Pixal3D] vendor/ was built for torch %s / %s / %s but "
                "this env is torch %s / %s / %s. Re-run `python install.py` on "
                "this machine.",
                _m.get("torch"), _m.get("cuda"), _m.get("python"),
                _torch_now, _cuda_now, _py_now,
            )
    except Exception:
        pass

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import (
        NODE_CLASS_MAPPINGS as _mappings,
        NODE_DISPLAY_NAME_MAPPINGS as _display,
    )

    NODE_CLASS_MAPPINGS.update(_mappings)
    NODE_DISPLAY_NAME_MAPPINGS.update(_display)
    log.info("[ComfyUI-Pixal3D] registered %d nodes", len(NODE_CLASS_MAPPINGS))
except Exception:
    log.error(
        "[ComfyUI-Pixal3D] failed to import nodes. If this mentions a missing "
        "package, run:  python custom_nodes/ComfyUI-Pixal3D/install.py\n%s",
        traceback.format_exc(),
    )

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
