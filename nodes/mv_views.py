"""Multi-view input for Pixal3D's MV pipeline (upstream TencentARC/Pixal3D f7cf384).

Turns V views + a camera rig into the bundle `Pixal3DMVImageTo3DPipeline.run_mv`
consumes, and checks the rig against the view silhouettes before the
multi-minute cascade runs.

Conventions (upstream inference_mv.py, Blender/NeRF): Z-up world, 4x4
camera-to-world matrices, each camera looks along its own -Z with +Y up, and
`camera_angle_x` is the horizontal FOV (radians) of a SQUARE frame. Frame 0 is
the main view and must be the canonical front view (camera at (0, -d, 0)): the
extractor re-bases every view onto it via calc_mat_i = F @ inv(C_0) @ C_i. An
azimuth of +90 puts the camera at +X, which sees the object's LEFT side -- the
side on the RIGHT of the front image.

Pure torch/numpy/PIL -- no comfy or pixal3d imports -- so the camera math is
testable off-GPU. `project()` and `relative_calc_mats()` mirror the vendored
`project_points_to_image_batch` / `compute_relative_calc_mat`; the
tests/test_mv_views.py suite checks them against upstream's source.
"""

import json
import logging
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("pixal3d")

# Upstream's example rig frames the unit cube at 1/1.1 of the frame:
# distance = 1.1 * 0.5 / tan(fov/2), i.e. 3.1192 at fov 20 deg.
UPSTREAM_RIG_MARGIN = 1.1
# SS / shape-512 stages condition at 512, the two 1024 stages at 1024.
COND_SIZES = (512, 1024)
# Below this silhouette coverage the rig and the views disagree (see rig_consistency).
COVERAGE_WARN = 0.85
# A competing rig must beat the chosen one's mean coverage by this much to be reported.
ALTERNATIVE_MARGIN = 0.02

DIRECTIONS = ("front_image_right_side", "front_image_left_side")
FRAMINGS = ("auto_fit", "upstream_rig", "manual")
ALIGNS = ("bbox_height", "none")
# bbox_height alignment: square canvas, widest object bbox at 1/ALIGN_MARGIN of it.
ALIGN_CANVAS = 1024
ALIGN_MARGIN = 1.15


# ============================================================================
# Camera math
# ============================================================================

def front_view_c2w(distance: float) -> torch.Tensor:
    """Canonical front view F (ProjGrid.front_view_transform_matrix at `distance`)."""
    return torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -float(distance)],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def orbit_c2w(azimuths_deg: Sequence[float], elevations_deg: Sequence[float],
              distance: float) -> torch.Tensor:
    """[V, 4, 4] c2w for cameras on a sphere of radius `distance` looking at the origin.

    azimuth 0 / elevation 0 is the front view (0, -d, 0); +azimuth turns toward +X.
    Reproduces upstream's assets/mv_images/example/transforms.json exactly.
    """
    az = torch.deg2rad(torch.tensor(azimuths_deg, dtype=torch.float64))
    el = torch.deg2rad(torch.tensor(elevations_deg, dtype=torch.float64))
    back = torch.stack([torch.sin(az) * torch.cos(el), -torch.cos(az) * torch.cos(el), torch.sin(el)], -1)
    # world_up x back, normalized; stays defined straight above/below the object
    right = torch.stack([torch.cos(az), torch.sin(az), torch.zeros_like(az)], -1)
    up = torch.cross(back, right, dim=-1)
    c2w = torch.eye(4, dtype=torch.float64).repeat(len(az), 1, 1)
    c2w[:, :3, 0] = right
    c2w[:, :3, 1] = up
    c2w[:, :3, 2] = back
    c2w[:, :3, 3] = back * float(distance)
    return c2w.float()


def relative_calc_mats(c2w: torch.Tensor) -> torch.Tensor:
    """calc_mat_i = F @ inv(C_0) @ C_i, F at view 0's distance (compute_relative_calc_mat)."""
    c2w = c2w.double()
    d0 = float(torch.norm(c2w[0, :3, 3]))
    rebase = front_view_c2w(d0).double() @ torch.linalg.inv(c2w[0])
    return (rebase[None] @ c2w).float()


def project(points: torch.Tensor, calc: torch.Tensor, fov: torch.Tensor,
            resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world points [N, 3] through V cameras -> (pixel xy [V, N, 2], depth [V, N]).

    Same math as project_points_to_image_batch: Blender pinhole with a 32 mm
    sensor, pixel-index coordinates, y flipped.
    """
    w2c = torch.linalg.inv(calc.double()).float()
    pts_h = torch.cat([points, torch.ones_like(points[:, :1])], -1)
    cam = torch.einsum("vij,nj->vni", w2c, pts_h)[..., :3]
    f_px = (16.0 / torch.tan(fov / 2.0) * resolution / 32.0)[:, None]
    z = -cam[..., 2] + 1e-8
    x = f_px * cam[..., 0] / z + resolution / 2.0
    y = -f_px * cam[..., 1] / z + resolution / 2.0
    return torch.stack([x, y], -1), -cam[..., 2]


def rig_distance(fov_rad: float, mesh_scale: float = 1.0) -> float:
    """Upstream example rig: the unit cube spans 1/1.1 of the frame."""
    return UPSTREAM_RIG_MARGIN * 0.5 / (math.tan(fov_rad / 2.0) * mesh_scale)


def fit_distance(alphas: Sequence[np.ndarray], fovs_rad: Sequence[float],
                 margin: float, mesh_scale: float = 1.0) -> float:
    """Camera distance at which the silhouettes fill 1/margin of the voxel grid.

    A silhouette reaching fraction r of the half-frame at distance d spans
    r * d * tan(fov/2) world units on the plane through the orbit center, so the
    widest view across the rig fixes d. Measured from the frame CENTER (the
    orbit axis), never from each view's own bbox: the views must share one scale.
    """
    widest = max(silhouette_radius(a) * math.tan(f / 2.0) for a, f in zip(alphas, fovs_rad))
    if widest <= 0:
        raise ValueError("Pixal3D multi-view: every view's mask is empty -- check the mask inputs.")
    return 0.5 / (margin * widest * mesh_scale)


def silhouette_radius(alpha: np.ndarray, thresh: float = 0.5) -> float:
    """Max |offset| of any foreground pixel from the frame center, as a fraction of the half-frame."""
    ys, xs = np.nonzero(alpha > thresh)
    if len(xs) == 0:
        return 0.0
    half = alpha.shape[0] / 2.0
    dx = np.maximum(np.abs(xs - half), np.abs(xs + 1 - half))
    dy = np.maximum(np.abs(ys - half), np.abs(ys + 1 - half))
    return float(max(dx.max(), dy.max()) / half)


def parse_angles(text: str, n: int, name: str, allow_auto: bool = False) -> List[float]:
    """'0,90,180,270' -> floats. One value broadcasts; 'auto' = n evenly spaced over 360."""
    t = (text or "").strip().lower()
    if allow_auto and t in ("", "auto"):
        return [i * 360.0 / n for i in range(n)]
    try:
        vals = [float(v) for v in t.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        raise ValueError(f"Pixal3D multi-view: can't parse {name} '{text}' -- use comma-separated degrees.")
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"Pixal3D multi-view: {name} has {len(vals)} values but there are {n} views.")
    return vals


# ============================================================================
# Images
# ============================================================================

def comfy_views(images: torch.Tensor, masks: Optional[torch.Tensor] = None,
                invert_mask: bool = False) -> List[Tuple[np.ndarray, np.ndarray]]:
    """ComfyUI IMAGE [V,H,W,3|4] (+ MASK [V,H,W]) -> [(rgb [H,W,3], alpha [H,W])] float32 in [0, 1].

    Alpha comes from `masks` (1.0 = object) or, failing that, the image's own
    alpha channel. There is no background removal here: without an object mask
    the background would be projected into the voxel grid.
    """
    if images.ndim == 3:
        images = images[None]
    V, H, W, C = images.shape
    if masks is not None:
        if masks.ndim == 2:
            masks = masks[None]
        if masks.shape[0] != V:
            raise ValueError(f"Pixal3D multi-view: {masks.shape[0]} masks for {V} views -- need one mask per view.")
        masks = masks.float()
        if invert_mask:
            masks = 1.0 - masks
        if masks.shape[1:] != (H, W):
            masks = F.interpolate(masks[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        alphas = masks
    elif C == 4 and not bool((images[..., 3] >= 254.0 / 255.0).all()):
        alphas = images[..., 3]
    else:
        raise ValueError(
            "Pixal3D multi-view: no object mask. Wire a MASK batch from a background-removal "
            "node (or LoadImage's MASK through InvertMask), or feed RGBA images with transparency.")
    rgb = images[..., :3].detach().cpu().float().clamp(0, 1).numpy()
    a = alphas.detach().cpu().float().clamp(0, 1).numpy()
    return [(rgb[i], a[i]) for i in range(V)]


def pad_to_square(rgb: np.ndarray, alpha: np.ndarray, fov_x: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Center-pad to square with transparent pixels; return the square frame's horizontal FOV.

    Padding keeps the focal length, so only a tall frame (padded left/right)
    widens the horizontal FOV. Upstream instead squashes non-square views, which
    breaks the projection.
    """
    H, W = alpha.shape
    if H == W:
        return rgb, alpha, fov_x
    S = max(H, W)
    y0, x0 = (S - H) // 2, (S - W) // 2
    rgb_sq = np.zeros((S, S, 3), dtype=np.float32)
    a_sq = np.zeros((S, S), dtype=np.float32)
    rgb_sq[y0:y0 + H, x0:x0 + W] = rgb
    a_sq[y0:y0 + H, x0:x0 + W] = alpha
    return rgb_sq, a_sq, 2.0 * math.atan(math.tan(fov_x / 2.0) * S / W)


def object_bbox(alpha: np.ndarray, thresh: float = 0.5) -> Tuple[int, int, int, int]:
    """(y0, y1, x0, x1) half-open bbox of the object; specks under 1% of the largest blob are ignored."""
    m = alpha > thresh
    if not m.any():
        raise ValueError("Pixal3D multi-view: a view's mask is empty -- check the mask inputs.")
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = np.bincount(lab.ravel())[1:]
            m = np.isin(lab, np.nonzero(sizes >= 0.01 * sizes.max())[0] + 1)
    except ImportError:
        pass
    ys, xs = np.nonzero(m)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def align_by_height(views: Sequence[Tuple[np.ndarray, np.ndarray]], fov_x: float,
                    size: int = ALIGN_CANVAS, margin: float = ALIGN_MARGIN):
    """Rescale + recenter every view so the object's bbox height matches, on one square canvas.

    On an eye-level orbit the object's vertical extent looks the same from every
    azimuth, so equal bbox heights restore one shared scale across views that were
    cropped or resized differently, and centering each bbox puts the orbit axis
    through the object's bbox center (exact for 90-degree steps). Perspective makes
    heights differ by a few percent per view, so frames that already share one
    camera are more exact with align_views=none.

    Returns (aligned [(rgb, alpha)], horizontal FOV of the canvas, notes). The canvas
    keeps view 0's focal length (scaled with it), so `fov_x` is view 0's FOV as given.
    """
    boxes = [object_bbox(a) for _, a in views]
    notes = [f"view {i}: the object is cut off at the top/bottom of its frame, so its height can't be matched"
             for i, ((_, a), (y0, y1, _x0, _x1)) in enumerate(zip(views, boxes)) if y0 == 0 or y1 == a.shape[0]]
    widest = max((x1 - x0) / (y1 - y0) for y0, y1, x0, x1 in boxes)
    target_h = size / (margin * max(1.0, widest))
    out = []
    for (rgb, a), (y0, y1, x0, x1) in zip(views, boxes):
        s = target_h / (y1 - y0)
        H, W = a.shape
        rgba = np.concatenate([rgb, a[..., None]], -1)
        img = Image.fromarray((rgba * 255.0).round().astype(np.uint8), mode="RGBA")
        img = img.resize((max(1, round(W * s)), max(1, round(H * s))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        canvas.paste(img, (round(size / 2 - (x0 + x1) / 2 * s), round(size / 2 - (y0 + y1) / 2 * s)))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        out.append((arr[..., :3], arr[..., 3]))
    s0 = target_h / (boxes[0][1] - boxes[0][0])
    f_px = views[0][1].shape[1] / 2.0 / math.tan(fov_x / 2.0) * s0
    return out, 2.0 * math.atan(size / 2.0 / f_px), notes


def cond_tensor(rgb: np.ndarray, alpha: np.ndarray, size: int) -> torch.Tensor:
    """inference_mv.to_cond_tensor: LANCZOS resize the RGBA view, premultiply -> [3, size, size]."""
    rgba = np.concatenate([rgb, alpha[..., None]], -1)
    img = Image.fromarray((rgba * 255.0).round().astype(np.uint8), mode="RGBA")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    a = torch.tensor(np.array(img.getchannel(3))).float() / 255.0
    c = torch.tensor(np.array(img.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    return c * a[None]


# ============================================================================
# Rig check + preview
# ============================================================================

def rig_consistency(alphas: Sequence[np.ndarray], calc: torch.Tensor, fov: torch.Tensor,
                    mesh_scale: float, grid_res: int = 128, check_res: int = 112):
    """Carve a visual hull from the silhouettes through the rig, reproject it into each view.

    Returns (coverage [V], hull_cover [V,R,R] bool, silhouettes [V,R,R] bool).
    Every hull voxel lies inside every silhouette by construction, so coverage =
    |reprojected hull| / |silhouette|. A correct rig explains ~all of each
    silhouette; a wrong FOV, framing, elevation or rotation direction carves the
    hull away and coverage drops in the views that disagree.
    """
    V = len(alphas)
    sil = torch.stack([torch.from_numpy(np.ascontiguousarray(a)).float() for a in alphas])
    sil = F.interpolate(sil[:, None], size=(check_res, check_res), mode="area")[:, 0] > 0.5
    lin = torch.linspace(-1.0, 1.0, grid_res)
    pts = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3) / mesh_scale / 2.0
    hull = torch.ones(pts.shape[0], dtype=torch.bool)
    for v in range(V):
        xy, depth = project(pts, calc[v:v + 1], fov[v:v + 1], check_res)
        norm = (xy[0] + 0.5) / check_res * 2.0 - 1.0          # ProjGrid's grid_sample normalization
        inside = (norm.abs() <= 1.0).all(-1) & (depth[0] > 0)
        hit = F.grid_sample(sil[v].float()[None, None], norm[None, :, None, :], mode="nearest",
                            align_corners=False, padding_mode="zeros")[0, 0, :, 0] > 0.5
        hull &= inside & hit
    cover = torch.zeros(V, check_res, check_res, dtype=torch.bool)
    hull_pts = pts[hull]
    coverage = torch.zeros(V)
    if hull_pts.shape[0]:
        for v in range(V):
            xy, _ = project(hull_pts, calc[v:v + 1], fov[v:v + 1], check_res)
            ix = (xy[0, :, 0] + 0.5).floor().long().clamp(0, check_res - 1)
            iy = (xy[0, :, 1] + 0.5).floor().long().clamp(0, check_res - 1)
            cover[v, iy, ix] = True
    for v in range(V):
        n_sil = int(sil[v].sum())
        coverage[v] = float((cover[v] & sil[v]).sum()) / max(n_sil, 1)
    return coverage, cover, sil


def _cube_edges(mesh_scale: float):
    h = 0.5 / mesh_scale
    corners = torch.tensor([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if int((corners[i] != corners[j]).sum()) == 1]
    return corners, edges


def render_preview(rgbs: Sequence[np.ndarray], alphas: Sequence[np.ndarray], calc: torch.Tensor,
                   fov: torch.Tensor, mesh_scale: float, cover: torch.Tensor, sil: torch.Tensor,
                   labels: Sequence[str], size: int = 512) -> torch.Tensor:
    """[V, size, size, 3] preview: premultiplied view, silhouette the rig can't explain in
    red, voxel-grid bounds in yellow, the grid's +X face (object's left) in green."""
    corners, edges = _cube_edges(mesh_scale)
    cxy, cdepth = project(corners, calc, fov, size)
    try:
        font = ImageFont.load_default(size=max(12, size // 28))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    out = []
    for v, (rgb, a) in enumerate(zip(rgbs, alphas)):
        base = cond_tensor(rgb, a, size).permute(1, 2, 0).numpy()
        miss = F.interpolate((sil[v] & ~cover[v]).float()[None, None], size=(size, size), mode="nearest")[0, 0].numpy() > 0.5
        base = base * 0.85
        base[miss] = base[miss] * 0.35 + np.array([0.65, 0.0, 0.0])
        img = Image.fromarray((base.clip(0, 1) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        if bool((cdepth[v] > 0).all()):
            for i, j in edges:
                on_left = float(corners[i, 0]) > 0 and float(corners[j, 0]) > 0
                draw.line([tuple(cxy[v, i].tolist()), tuple(cxy[v, j].tolist())],
                          fill=(60, 220, 60) if on_left else (235, 200, 40), width=2)
        draw.text((8, 6), labels[v], fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        out.append(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0))
    return torch.stack(out)


# ============================================================================
# Bundle
# ============================================================================

def finalize(rgbs: Sequence[np.ndarray], alphas: Sequence[np.ndarray], c2w: torch.Tensor,
             fovs_rad: Sequence[float], mesh_scale: float, names: Sequence[str],
             alternatives: Optional[dict] = None, notes: Sequence[str] = (),
             preview_size: int = 512):
    """Square RGBA views + c2w [V,4,4] + per-view FOVs -> (bundle, preview [V,S,S,3], report str).

    The bundle is exactly what inference_mv.load_views builds for run_mv.
    `alternatives` maps a hint ("set X to Y") to a competing c2w rig; if one
    explains the silhouettes clearly better, the report says so. This catches
    errors that a coverage threshold alone misses on near-symmetric objects.
    """
    V = len(rgbs)
    fov = torch.tensor(list(fovs_rad), dtype=torch.float32)
    calc = relative_calc_mats(c2w)
    notes = list(notes)

    d0 = float(torch.norm(c2w[0, :3, 3]))
    front_err = float((c2w[0] - front_view_c2w(d0)).abs().max())
    if front_err > 1e-4:
        notes.append(
            f"view 0 is not the canonical eye-level front view (max deviation {front_err:.3f}): "
            f"the mesh comes out posed in view 0's camera frame, e.g. tilted by its elevation")
    if d0 * mesh_scale < math.sqrt(3) / 2:
        notes.append(f"camera distance {d0:.3f} is inside the voxel grid -- the FOV is probably too wide")
    for i, a in enumerate(alphas):
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        if (border > 0.5).any():
            notes.append(f"view {i} ({names[i]}): the object touches the frame edge -- it must be fully in frame")
        ys, xs = np.nonzero(a > 0.5)
        if len(xs):
            half = a.shape[0] / 2.0
            off = max(abs((xs.min() + xs.max() + 1) / 2 - half), abs((ys.min() + ys.max() + 1) / 2 - half)) / a.shape[0]
            if off > 0.15:
                notes.append(f"view {i} ({names[i]}): object is off-center by {off:.0%} of the frame -- "
                             f"views must share the orbit center (never crop views individually)")

    coverage, cover, sil = rig_consistency(alphas, calc, fov, mesh_scale)
    bad = [i for i in range(V) if coverage[i] < COVERAGE_WARN]
    if bad:
        notes.append(
            "silhouette coverage below "
            f"{COVERAGE_WARN:.0%} in view(s) {', '.join(str(i) for i in bad)} -- the rig doesn't match "
            "the images (rotation direction, FOV, elevation or framing). Red areas in the preview "
            "are what the rig can't explain")

    for hint, alt_c2w in (alternatives or {}).items():
        alt_cov, _, _ = rig_consistency(alphas, relative_calc_mats(alt_c2w), fov, mesh_scale)
        if float(alt_cov.mean()) > float(coverage.mean()) + ALTERNATIVE_MARGIN:
            notes.append(
                f"the rig with {hint} explains the silhouettes better "
                f"(mean coverage {float(alt_cov.mean()):.0%} vs {float(coverage.mean()):.0%})")

    labels = [f"#{i} {names[i]}  fov {math.degrees(float(fov[i])):.1f}  cov {float(coverage[i]):.0%}"
              for i in range(V)]
    preview = render_preview(rgbs, alphas, calc, fov, mesh_scale, cover, sil, labels, preview_size)

    transform_matrix = c2w[None].float()
    bundle = {
        "images": {size: torch.stack([cond_tensor(r, a, size) for r, a in zip(rgbs, alphas)])[None]
                   for size in COND_SIZES},                                       # [1, V, 3, S, S]
        "camera_angle_x": fov[None],                                              # [1, V]
        "camera_distance": torch.norm(transform_matrix[:, :, :3, 3], dim=-1),     # [1, V]
        "transform_matrix": transform_matrix,                                     # [1, V, 4, 4]
        "mesh_scale": float(mesh_scale),
        "view_names": list(names),
    }
    report = (f"V={V} views, distance {d0:.4f}, mesh_scale {mesh_scale:.3f}, "
              f"coverage min {float(coverage.min()):.0%} / mean {float(coverage.mean()):.0%}")
    for n in notes:
        log.warning(f"[pixal3d-mv] {n}")
    log.info(f"[pixal3d-mv] {report}")
    return bundle, preview, "\n".join([report] + [f"WARNING: {n}" for n in notes])


def views_from_orbit(images: torch.Tensor, masks: Optional[torch.Tensor], invert_mask: bool,
                     azimuths: str, elevations: str, fov_deg: float, direction: str,
                     front_index: int, framing: str, margin: float, distance: float,
                     mesh_scale: float, align: str = "bbox_height"):
    """Orbit rig (turntable / multi-view generator frames) -> finalize() outputs."""
    views = comfy_views(images, masks, invert_mask)
    V = len(views)
    az = parse_angles(azimuths, V, "azimuths", allow_auto=True)
    el = parse_angles(elevations, V, "elevations")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    if direction == "front_image_left_side":
        az = [-a for a in az]
    if not 0 <= front_index < V:
        raise ValueError(f"front_index {front_index} out of range for {V} views")
    order = [(front_index + i) % V for i in range(V)]
    views, az, el = [views[i] for i in order], [az[i] for i in order], [el[i] for i in order]
    az = [a - az[0] for a in az]
    names = [f"f{order[i]} az{az[i] % 360:.0f} el{el[i]:.0f}" for i in range(V)]

    fov_in = math.radians(fov_deg)
    notes = []
    if align == "bbox_height":
        if any(abs(e) > 1e-6 for e in el):
            notes.append("align_views=bbox_height assumes an eye-level orbit; with nonzero elevation the "
                         "object's height legitimately differs per view -- use align_views=none")
        aligned, fov_canvas, align_notes = align_by_height(views, fov_in)
        notes += align_notes
        rgbs = [v[0] for v in aligned]
        alphas = [v[1] for v in aligned]
        fovs = [fov_canvas] * V
    elif align == "none":
        squared = [pad_to_square(rgb, a, fov_in) for rgb, a in views]
        rgbs = [s[0] for s in squared]
        alphas = [s[1] for s in squared]
        fovs = [s[2] for s in squared]
    else:
        raise ValueError(f"align_views must be one of {ALIGNS}")

    if framing == "auto_fit":
        d = fit_distance(alphas, fovs, margin, mesh_scale)
    elif framing == "upstream_rig":
        d = rig_distance(fovs[0], mesh_scale)
    elif framing == "manual":
        d = float(distance)
    else:
        raise ValueError(f"framing must be one of {FRAMINGS}")
    other = DIRECTIONS[1 - DIRECTIONS.index(direction)]
    mirrored = {f"view_at_plus_90 = {other}": orbit_c2w([-a for a in az], el, d)}
    return finalize(rgbs, alphas, orbit_c2w(az, el, d), fovs, mesh_scale, names,
                    alternatives=mirrored, notes=notes)


def views_from_dir(views_dir: str, num_views: int = 0):
    """Upstream dataset-style folder: transforms.json + RGBA views -> finalize() outputs."""
    with open(os.path.join(views_dir, "transforms.json")) as f:
        meta = json.load(f)
    frames = meta["frames"]
    if num_views:
        if num_views > len(frames):
            raise ValueError(f"num_views {num_views} > {len(frames)} views in {views_dir}")
        frames = frames[:num_views]
    rgbs, alphas, fovs, names = [], [], [], []
    for fr in frames:
        fov = fr.get("camera_angle_x", meta.get("camera_angle_x"))
        if fov is None:
            raise KeyError(f"'camera_angle_x' missing for {fr.get('file_path')}")
        img = Image.open(os.path.join(views_dir, fr["file_path"]))
        if img.mode != "RGBA" or np.all(np.array(img.getchannel(3)) == 255):
            raise ValueError(f"{fr['file_path']}: needs an alpha channel (no background removal in this loader)")
        arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        rgb, a, fov_sq = pad_to_square(arr[..., :3], arr[..., 3], float(fov))
        rgbs.append(rgb)
        alphas.append(a)
        fovs.append(fov_sq)
        names.append(fr.get("name", os.path.splitext(fr["file_path"])[0]))
    c2w = torch.tensor([fr["transform_matrix"] for fr in frames], dtype=torch.float32)
    return finalize(rgbs, alphas, c2w, fovs, float(meta.get("mesh_scale", 1.0)), names)
