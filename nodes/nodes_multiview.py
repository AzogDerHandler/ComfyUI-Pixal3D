"""Pixal3D multi-view nodes (upstream TencentARC/Pixal3D f7cf384, `*_mv` weights).

Workflow:
    Pixal3DMultiViewInput        IMAGE batch (+ MASK batch) + orbit rig -> PIXAL3D_MV_VIEWS + preview
      or Pixal3DLoadMultiViewFolder  transforms.json folder             -> PIXAL3D_MV_VIEWS + preview
    Pixal3DGenerateMeshMV        -> (TRIMESH, PIXAL3D_VOXELGRID) -> the same ProcessMesh ->
                                    RasterizePBR -> ExportGLB chain as single-view
      or Pixal3DGenerateGLBMV    -> vertex-color GLB (the monolithic convenience path)

PIXAL3D_MV_VIEWS is the dict `Pixal3DMVImageTo3DPipeline.run_mv` consumes
(see mv_views.finalize). The multi-view pipeline shares the decoders, DINOv3
and NAF with single-view; only its four flow DiTs (~22 GB on disk) are extra,
downloaded on first use.
"""

import logging
import os

from comfy_api.latest import io

from . import mv_views

log = logging.getLogger("pixal3d")

_BUNDLED_EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "mv_example")


def _views_outputs():
    return [
        io.Custom("PIXAL3D_MV_VIEWS").Output(display_name="views"),
        io.Image.Output(
            display_name="preview",
            tooltip=(
                "Each view as the model sees it (premultiplied, square). Yellow: the voxel "
                "grid's outline at the object's depth -- the object must fit inside; green: its "
                "+X side (object's left). Red: silhouette the rig can't explain -- wrong "
                "rotation direction, FOV, elevation or framing."
            ),
        ),
        io.String.Output(display_name="report"),
    ]


def _cascade_inputs():
    """Seed + token budget + per-stage sampler knobs, identical to the single-view nodes."""
    return [
        io.Int.Input("seed", default=42, min=0, max=2**31 - 1),
        io.Int.Input(
            "max_num_tokens", default=49152, min=1024, max=262144, step=1024, optional=True,
            tooltip=(
                "Sparse token budget for the HR stages -- the main geometry-"
                "detail lever. The cascade auto-shrinks resolution until the "
                "budget fits (check logs). 49152 = upstream default; "
                "98304-131072 is a reasonable max-quality range on big GPUs."
            ),
        ),
        io.Int.Input("ss_steps", default=12, min=1, max=64, optional=True),
        io.Float.Input("ss_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True),
        io.Float.Input("ss_rescale", default=0.7, min=0.0, max=1.0, step=0.05, optional=True),
        io.Float.Input("ss_rescale_t", default=5.0, min=0.0, max=10.0, step=0.1, optional=True),
        io.Int.Input("shape_steps", default=12, min=1, max=64, optional=True),
        io.Float.Input("shape_guidance", default=7.5, min=0.0, max=15.0, step=0.1, optional=True),
        io.Float.Input("shape_rescale", default=0.5, min=0.0, max=1.0, step=0.05, optional=True),
        io.Float.Input("shape_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True),
        io.Int.Input("tex_steps", default=12, min=1, max=64, optional=True),
        io.Float.Input("tex_guidance", default=1.0, min=0.0, max=15.0, step=0.1, optional=True),
        io.Float.Input("tex_rescale", default=0.0, min=0.0, max=1.0, step=0.05, optional=True),
        io.Float.Input("tex_rescale_t", default=3.0, min=0.0, max=10.0, step=0.1, optional=True),
    ]


class Pixal3DMultiViewInput(io.ComfyNode):
    """Frames of an orbit (turntable video, multi-view generator) -> PIXAL3D_MV_VIEWS."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DMultiViewInput",
            display_name="Pixal3D Multi-View Input",
            category="Pixal3D",
            description=(
                "Builds the multi-view bundle from an IMAGE batch of views around the object "
                "(e.g. 4 or 8 frames of an orbit video) plus a camera rig given as azimuths / "
                "elevations / FOV. By default every view is rescaled so the object's bbox height "
                "matches and recentered (align_views=bbox_height), which repairs frames of mixed "
                "sizes / crops on an eye-level orbit. Check the preview + report before running "
                "the cascade."
            ),
            inputs=[
                io.Image.Input("images", tooltip="All views as one batch, in orbit order. RGBA works as-is."),
                io.Mask.Input(
                    "masks", optional=True,
                    tooltip=(
                        "Object masks, one per view (1.0 = object), e.g. a background-removal "
                        "node run on the batch. Required unless the images carry alpha."
                    ),
                ),
                io.Boolean.Input(
                    "invert_mask", default=False, optional=True,
                    tooltip="Enable for LoadImage's MASK output, which is 1.0 where the image is transparent.",
                ),
                io.Float.Input(
                    "fov_deg", default=20.0, min=1.0, max=120.0, step=0.1,
                    tooltip=(
                        "Horizontal FOV of the front frame as given, in degrees. 20 for upstream-style "
                        "renders; for video frames wire Pixal3D Estimate Camera's fov_x_deg run on the "
                        "front frame."
                    ),
                ),
                io.String.Input(
                    "azimuths", default="auto",
                    tooltip=(
                        "Camera azimuth per view in degrees, comma-separated, orbit order. "
                        "'auto' = evenly spaced over 360 (4 views -> 0,90,180,270; 8 -> every 45)."
                    ),
                ),
                io.String.Input(
                    "elevations", default="0",
                    tooltip="Camera elevation per view in degrees (one value = all views). Positive = looking down.",
                ),
                io.Combo.Input(
                    "view_at_plus_90", options=list(mv_views.DIRECTIONS), default=mv_views.DIRECTIONS[0],
                    tooltip=(
                        "Which side of the object the view at +90 deg shows, named by where that side "
                        "sits in the FRONT image. front_image_right_side is upstream's convention "
                        "(camera moves toward +X, sees the object's left). The report flags it if the "
                        "other direction fits the silhouettes better."
                    ),
                ),
                io.Int.Input(
                    "front_index", default=0, min=0, max=63,
                    tooltip=(
                        "Which frame is the front view. The model poses the mesh in that view's frame; "
                        "an eye-level front view gives an upright mesh."
                    ),
                ),
                io.Combo.Input(
                    "framing", options=list(mv_views.FRAMINGS), default="auto_fit",
                    tooltip=(
                        "auto_fit: camera distance from the silhouettes so the object fills "
                        "1/grid_fill_margin of the voxel grid (matches single-view's framing). "
                        "upstream_rig: the unit cube spans 1/1.1 of the frame (upstream's example "
                        "renders and rig-framed generators). manual: use `distance`."
                    ),
                ),
                io.Float.Input("grid_fill_margin", default=1.1, min=1.0, max=2.0, step=0.01, optional=True),
                io.Float.Input("distance", default=3.0, min=0.1, max=100.0, step=0.01, optional=True,
                               tooltip="Camera distance for framing=manual (voxel grid spans [-0.5, 0.5])."),
                io.Float.Input("mesh_scale", default=1.0, min=0.1, max=10.0, step=0.05, optional=True),
                # Last on purpose: saved workflows fill widgets by position.
                io.Combo.Input(
                    "align_views", options=list(mv_views.ALIGNS), default="bbox_height", optional=True,
                    tooltip=(
                        "bbox_height: rescale + recenter each view so the object's height matches -- on "
                        "an eye-level orbit the height is the same from every side, so this fixes frames "
                        "of different sizes or crops. none: use the frames exactly as given (only pads "
                        "to square) -- more exact for frames that already share one camera, and required "
                        "for elevated orbits."
                    ),
                ),
            ],
            outputs=_views_outputs(),
        )

    @classmethod
    def execute(
        cls,
        images,
        fov_deg: float,
        azimuths: str,
        elevations: str,
        view_at_plus_90: str,
        front_index: int,
        framing: str,
        align_views: str = "bbox_height",
        masks=None,
        invert_mask: bool = False,
        grid_fill_margin: float = 1.1,
        distance: float = 3.0,
        mesh_scale: float = 1.0,
    ):
        from .stages import _phase
        with _phase("Pixal3DMultiViewInput.execute"):
            views, preview, report = mv_views.views_from_orbit(
                images, masks, invert_mask, azimuths, elevations, fov_deg, view_at_plus_90,
                front_index, framing, grid_fill_margin, distance, mesh_scale, align=align_views,
            )
            return io.NodeOutput(views, preview, report)


class Pixal3DLoadMultiViewFolder(io.ComfyNode):
    """Upstream dataset-style folder (transforms.json + RGBA views) -> PIXAL3D_MV_VIEWS."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DLoadMultiViewFolder",
            display_name="Pixal3D Load Multi-View Folder",
            category="Pixal3D",
            description=(
                "Loads a folder in upstream's inference_mv.py format: transforms.json (Blender/NeRF "
                "c2w, camera_angle_x) + RGBA views; frame 0 must be the front view. Empty path = the "
                "bundled upstream example (4 views), a known-good end-to-end test."
            ),
            inputs=[
                io.String.Input(
                    "folder_path", default="",
                    tooltip="Absolute, or relative to ComfyUI/input. Empty = bundled upstream example.",
                ),
                io.Int.Input("num_views", default=0, min=0, max=64, optional=True,
                             tooltip="Use only the first N frames (0 = all)."),
            ],
            outputs=_views_outputs(),
        )

    @classmethod
    def execute(cls, folder_path: str, num_views: int = 0):
        from .stages import _phase
        import folder_paths
        path = (folder_path or "").strip() or _BUNDLED_EXAMPLE
        if not os.path.isabs(path):
            path = os.path.join(folder_paths.get_input_directory(), path)
        if not os.path.isfile(os.path.join(path, "transforms.json")):
            raise FileNotFoundError(f"Pixal3D: no transforms.json in {path}")
        with _phase("Pixal3DLoadMultiViewFolder.execute"):
            views, preview, report = mv_views.views_from_dir(path, num_views)
            return io.NodeOutput(views, preview, report)


def _generate_kwargs(pipeline, views, kw):
    return dict(
        image=None,
        camera_params=None,
        views=views,
        pipeline_type=pipeline.get("pipeline_type", "1024_cascade"),
        attn_backend=pipeline.get("attn_backend", "auto"),
        vram_mode=pipeline.get("vram_mode", "auto"),
        **kw,
    )


class Pixal3DGenerateMeshMV(io.ComfyNode):
    """Multi-view cascade. Emits the raw DC mesh + the sparse PBR voxel grid."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DGenerateMeshMV",
            display_name="Pixal3D Generate Mesh (Multi-View)",
            category="Pixal3D",
            description=(
                "Runs the four-stage cascade with the multi-view weights, conditioned on all views "
                "at once. Same outputs as Pixal3D Generate Mesh -- pipe into Pixal3DProcessMesh + "
                "Pixal3DRasterizePBR + Pixal3DExportGLB. First use downloads the ~22 GB *_mv weights."
            ),
            inputs=[
                io.Custom("PIXAL3D_PIPELINE").Input("pipeline", tooltip="From Pixal3DLoadPipeline."),
                io.Custom("PIXAL3D_MV_VIEWS").Input("views", tooltip="From Pixal3D Multi-View Input / Load Folder."),
            ] + _cascade_inputs(),
            outputs=[
                io.Custom("TRIMESH").Output(display_name="mesh"),
                io.Custom("PIXAL3D_VOXELGRID").Output(display_name="voxelgrid"),
            ],
        )

    @classmethod
    def execute(cls, pipeline, views, **kw):
        from .stages import generate_mesh_and_voxelgrid, _YUP_TO_ZUP_ROT, _phase
        with _phase("Pixal3DGenerateMeshMV.execute"):
            tri, voxelgrid = generate_mesh_and_voxelgrid(**_generate_kwargs(pipeline, views, kw))
            # Same frame handling as Pixal3DGenerateMesh: downstream bake expects Z-up.
            tri.apply_transform(_YUP_TO_ZUP_ROT)
            log.info(
                f"[Pixal3DGenerateMeshMV] mesh={len(tri.vertices)} verts / {len(tri.faces)} faces, "
                f"voxelgrid={voxelgrid['attrs'].shape[0]} voxels"
            )
            return io.NodeOutput(tri, voxelgrid)


class Pixal3DGenerateGLBMV(io.ComfyNode):
    """Multi-view cascade + vertex-color GLB (the monolithic convenience path)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Pixal3DGenerateGLBMV",
            display_name="Pixal3D Generate GLB (Multi-View)",
            category="Pixal3D",
            is_output_node=True,
            description=(
                "Multi-view twin of Pixal3D Generate GLB: cascade + light cleanup + vertex colors. "
                "For UV-baked PBR textures use Generate Mesh (Multi-View) and the mesh chain."
            ),
            inputs=[
                io.Custom("PIXAL3D_PIPELINE").Input("pipeline", tooltip="From Pixal3DLoadPipeline."),
                io.Custom("PIXAL3D_MV_VIEWS").Input("views", tooltip="From Pixal3D Multi-View Input / Load Folder."),
            ] + _cascade_inputs() + [
                io.Int.Input("decimation_target", default=200000, min=10000, max=1000000, step=10000, optional=True),
                io.Int.Input("texture_size", default=2048, min=512, max=4096, step=256, optional=True),
                io.Boolean.Input("force_opaque", default=True, optional=True),
                io.Boolean.Input("double_sided", default=False, optional=True),
                io.Boolean.Input("remove_inner_faces", default=False, optional=True),
                io.String.Input("filename_prefix", default="pixal3d_mv", optional=True),
            ],
            outputs=[
                io.String.Output(display_name="glb_filepath"),
            ],
        )

    @classmethod
    def execute(cls, pipeline, views, **kw):
        from .stages import generate_glb, _phase
        with _phase("Pixal3DGenerateGLBMV.execute"):
            return io.NodeOutput(generate_glb(**_generate_kwargs(pipeline, views, kw)))


NODE_CLASS_MAPPINGS = {
    "Pixal3DMultiViewInput": Pixal3DMultiViewInput,
    "Pixal3DLoadMultiViewFolder": Pixal3DLoadMultiViewFolder,
    "Pixal3DGenerateMeshMV": Pixal3DGenerateMeshMV,
    "Pixal3DGenerateGLBMV": Pixal3DGenerateGLBMV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Pixal3DMultiViewInput": "Pixal3D Multi-View Input",
    "Pixal3DLoadMultiViewFolder": "Pixal3D Load Multi-View Folder",
    "Pixal3DGenerateMeshMV": "Pixal3D Generate Mesh (Multi-View)",
    "Pixal3DGenerateGLBMV": "Pixal3D Generate GLB (Multi-View)",
}
