"""Pixal3D loader nodes.

Pixal3DLoadPipeline triggers weight download + pipeline construction. It
returns a sentinel so it can act as an upstream dependency in the graph;
the actual pipeline lives in the module-level cache inside the isolation env.
"""

import logging

import torch
from comfy_api.latest import io

log = logging.getLogger("pixal3d")


class Pixal3DLoadPipeline(io.ComfyNode):
    """Load (and download on first run) the Pixal3D cascade pipeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DLoadPipeline",
            display_name="Load Pixal3D Pipeline",
            category="Pixal3D",
            description=(
                "Downloads ~22 GB of Pixal3D weights from HuggingFace on first run, "
                "then builds the cascade pipeline + four DinoV3 image-cond models. "
                "Returns a sentinel that downstream nodes depend on."
            ),
            inputs=[
                io.Combo.Input(
                    "pipeline_type",
                    options=["1024_cascade", "1536_cascade"],
                    default="1536_cascade",
                    tooltip=(
                        "Cascade target resolution. 1536_cascade is the max-quality "
                        "path (default here; needs ~24 GB+ VRAM). Drop to "
                        "1024_cascade on smaller cards."
                    ),
                ),
                io.Combo.Input(
                    "attn_backend",
                    options=["auto", "flash_attn", "flash_attn_3", "sdpa", "xformers", "naive"],
                    default="auto",
                    tooltip=(
                        "Dense + sparse attention backend (pixal3d native dispatch). "
                        "'auto' probes flash_attn_3 -> flash_attn -> xformers -> sdpa. "
                        "'flash_attn_3' needs the separate flash_attn_interface package. "
                        "Note: sageattention is not in pixal3d's native dispatch."
                    ),
                    optional=True,
                ),
                io.Combo.Input(
                    "vram_mode",
                    options=["auto", "full_gpu", "low_vram"],
                    default="auto",
                    tooltip=(
                        "full_gpu: keep all 13 models resident on the GPU -- no "
                        "per-stage CPU<->GPU swapping, fastest runs (needs a big "
                        "card; made for cloud GPUs). low_vram: pixal3d's per-stage "
                        "swap, fits 24 GB. auto: full_gpu when total VRAM >= 30 GB."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("PIXAL3D_PIPELINE").Output(display_name="pipeline"),
            ],
        )

    @classmethod
    def execute(
        cls,
        pipeline_type: str = "1536_cascade",
        attn_backend: str = "auto",
        vram_mode: str = "auto",
    ):
        from .stages import _phase
        with _phase("Pixal3DLoadPipeline.execute"):
            # Thin: just emit a config dict. The actual model load is lazy and
            # fires from Pixal3DGenerateGLB (and Pixal3DPreprocessImage if the
            # workflow runs rembg). Matches TRELLIS2's LoadTrellis2Models pattern.
            return io.NodeOutput({
                "pipeline_type": pipeline_type,
                "attn_backend": attn_backend,
                "vram_mode": vram_mode,
            })


NODE_CLASS_MAPPINGS = {
    "Pixal3DLoadPipeline": Pixal3DLoadPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DLoadPipeline": "Load Pixal3D Pipeline",
}
