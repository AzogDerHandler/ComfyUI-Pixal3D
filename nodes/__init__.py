"""ComfyUI-Pixal3D nodes — aggregated NODE_CLASS_MAPPINGS for register_nodes()."""

# Note: pixal3d code now imports the installed wheel names directly --
# `cumesh_vb`, `o_voxel_vb_ap`, `flex_gemm_ap` (matching TRELLIS2's pattern).
# A previous sys.modules alias hack here aliased the bare names; that turned
# out to silently swallow ImportError and mask real failures, so it was
# removed and every call site was renamed to use the wheel names explicitly.

from .nodes_loader import (
    NODE_CLASS_MAPPINGS as loader_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as loader_display,
)
from .nodes_inference import (
    NODE_CLASS_MAPPINGS as inference_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as inference_display,
)
from .nodes_mesh import (
    NODE_CLASS_MAPPINGS as mesh_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as mesh_display,
)
from .nodes_multiview import (
    NODE_CLASS_MAPPINGS as multiview_mappings,
    NODE_DISPLAY_NAME_MAPPINGS as multiview_display,
)

NODE_CLASS_MAPPINGS = {
    **loader_mappings,
    **inference_mappings,
    **mesh_mappings,
    **multiview_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **loader_display,
    **inference_display,
    **mesh_display,
    **multiview_display,
}
