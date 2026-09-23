"""Off-GPU tests for nodes/mv_views.py (multi-view camera math + rig check).

Run: python -m pytest tests/test_mv_views.py   (or: python tests/test_mv_views.py)
Needs only torch, numpy, pillow -- mv_views imports nothing from comfy/pixal3d.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "assets" / "mv_example"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mv = _load("mv_views", ROOT / "nodes" / "mv_views.py")
ref = _load("upstream_ref", Path(__file__).with_name("upstream_ref.py"))


# ----------------------------------------------------------------------------
# Synthetic asymmetric object, ray-marched into exact silhouettes
# ----------------------------------------------------------------------------

def _inside(p):
    box = (p[..., 0].abs() <= 0.25) & (p[..., 1].abs() <= 0.12) & (p[..., 2].abs() <= 0.4)
    # a knob on the object's front-left-top (+X is the object's left, -Y its front)
    knob = ((p - torch.tensor([0.3, -0.1, 0.22])) ** 2).sum(-1) <= 0.12 ** 2
    return box | knob


def _render_silhouettes(c2w, fov, res=256, n_steps=320):
    """Pixel i's ray goes through pixel-index coordinate i, matching project()."""
    out = []
    ij = torch.arange(res, dtype=torch.float32)
    y, x = torch.meshgrid(ij, ij, indexing="ij")
    f_px = res / (2.0 * math.tan(fov / 2.0))
    d_cam = torch.stack([(x - res / 2.0) / f_px, -(y - res / 2.0) / f_px, -torch.ones_like(x)], -1)
    for C in c2w:
        dirs = d_cam.reshape(-1, 3) @ C[:3, :3].T
        origin = C[:3, 3]
        dist = float(origin.norm())
        t = torch.linspace(dist - 0.9, dist + 0.9, n_steps)
        hit = torch.zeros(dirs.shape[0], dtype=torch.bool)
        for chunk in torch.split(t, 64):
            pts = origin + dirs[:, None, :] * chunk[None, :, None]
            hit |= _inside(pts).any(1)
        out.append(hit.reshape(res, res).float().numpy())
    return out


def _as_comfy(alphas, gray=0.5):
    """Silhouettes -> ComfyUI IMAGE batch (flat gray object) + MASK batch."""
    a = torch.from_numpy(np.stack(alphas))
    img = a[..., None].repeat(1, 1, 1, 3) * gray
    return img, a


# ----------------------------------------------------------------------------
# Camera math vs upstream
# ----------------------------------------------------------------------------

def test_rig_distance_matches_upstream_example():
    assert abs(mv.rig_distance(math.radians(20.0)) - 3.1192049980163574) < 1e-5


def test_orbit_reproduces_upstream_transforms():
    meta = json.loads((EXAMPLE / "transforms.json").read_text())
    expected = torch.tensor([fr["transform_matrix"] for fr in meta["frames"]])
    got = mv.orbit_c2w([0, 90, 180, 270], [0] * 4, mv.rig_distance(meta["camera_angle_x"]))
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()


def test_calc_mats_and_projection_match_upstream():
    torch.manual_seed(0)
    for view0 in [(0.0, 0.0), (30.0, 15.0)]:  # canonical front view, and a re-based one
        az = [view0[0], 45.0, 160.0, 250.0, 300.0]
        el = [view0[1], 20.0, -10.0, 35.0, 80.0]
        c2w = mv.orbit_c2w(az, el, 2.7)
        fov = torch.tensor([0.35, 0.5, 0.7, 0.9, 1.1])
        dist = torch.norm(c2w[:, :3, 3], dim=-1)
        F0 = torch.tensor([[1.0, 0, 0, 0], [0, 0, -1, -2.0], [0, 1, 0, 0], [0, 0, 0, 1]])
        up_calc = ref.compute_relative_calc_mat(c2w[None], dist[None], F0)[0]
        calc = mv.relative_calc_mats(c2w)
        assert torch.allclose(calc, up_calc, atol=1e-5)
        pts = torch.rand(500, 3) - 0.5
        xy, depth = mv.project(pts, calc, fov, 512)
        up_xy, up_depth, _ = ref.project_points_to_image_batch(pts, up_calc, fov, 512)
        assert torch.allclose(xy, up_xy, atol=1e-2)
        assert torch.allclose(depth, up_depth, atol=1e-5)


def test_cond_tensor_matches_upstream():
    img = Image.open(EXAMPLE / "view01_azim090.png")
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    for size in mv.COND_SIZES:
        got = mv.cond_tensor(arr[..., :3], arr[..., 3], size)
        assert torch.equal(got, ref.to_cond_tensor(img.convert("RGBA"), size))


def test_parse_angles():
    assert mv.parse_angles("auto", 8, "az", allow_auto=True) == [i * 45.0 for i in range(8)]
    assert mv.parse_angles("15", 3, "el") == [15.0] * 3
    assert mv.parse_angles("0, 90;180,270", 4, "az") == [0, 90, 180, 270]
    try:
        mv.parse_angles("0,90", 4, "az")
        raise AssertionError("length mismatch not caught")
    except ValueError:
        pass


def test_pad_to_square_preserves_projection():
    fov = math.radians(30.0)
    rgb, a = np.zeros((300, 200, 3), np.float32), np.zeros((300, 200), np.float32)
    rgb_sq, a_sq, fov_sq = mv.pad_to_square(rgb, a, fov)           # tall: padded left/right
    assert a_sq.shape == (300, 300)
    assert abs(math.tan(fov_sq / 2) - math.tan(fov / 2) * 300 / 200) < 1e-9
    _, a_w, fov_w = mv.pad_to_square(np.zeros((200, 300, 3), np.float32), np.zeros((200, 300), np.float32), fov)
    assert a_w.shape == (300, 300) and fov_w == fov                  # wide: horizontal FOV unchanged


# ----------------------------------------------------------------------------
# End to end on views
# ----------------------------------------------------------------------------

def test_upstream_example_folder():
    bundle, preview, report = mv.views_from_dir(str(EXAMPLE))
    assert bundle["images"][512].shape == (1, 4, 3, 512, 512)
    assert bundle["images"][1024].shape == (1, 4, 3, 1024, 1024)
    assert bundle["camera_angle_x"].shape == (1, 4) and bundle["transform_matrix"].shape == (1, 4, 4, 4)
    assert abs(float(bundle["camera_distance"][0, 0]) - 3.1192049980163574) < 1e-5
    assert preview.shape == (4, 512, 512, 3)
    assert "WARNING" not in report, report
    print("upstream example:", report)


def test_orbit_node_path_equals_folder_path():
    """The IMAGE-batch orbit path with upstream's rig reproduces the folder bundle exactly."""
    frames = sorted(EXAMPLE.glob("view*.png"))
    arr = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGBA"), np.float32) / 255.0 for p in frames]))
    b_orbit, _, _ = mv.views_from_orbit(arr, None, False, "auto", "0", 20.0, "front_image_right_side",
                                        0, "upstream_rig", 1.1, 0.0, 1.0, align="none")
    b_dir, _, _ = mv.views_from_dir(str(EXAMPLE))
    assert torch.allclose(b_orbit["transform_matrix"], b_dir["transform_matrix"], atol=1e-5)
    for s in mv.COND_SIZES:
        assert torch.equal(b_orbit["images"][s], b_dir["images"][s])


def _synthetic(n_views=8, fov_deg=35.0, el=10.0, d=2.2, res=256):
    az = [i * 360.0 / n_views for i in range(n_views)]
    c2w = mv.orbit_c2w(az, [el] * n_views, d)
    return _render_silhouettes(c2w, math.radians(fov_deg), res)


def test_rig_check_accepts_correct_and_flags_flipped_direction():
    alphas = _synthetic()
    img, msk = _as_comfy(alphas)
    common = dict(masks=msk, invert_mask=False, azimuths="auto", elevations="10", front_index=0,
                  framing="manual", margin=1.1, distance=2.2, mesh_scale=1.0, fov_deg=35.0, align="none")
    _, _, good = mv.views_from_orbit(img, direction="front_image_right_side", **common)
    _, _, flipped = mv.views_from_orbit(img, direction="front_image_left_side", **common)
    print("correct:", good)
    print("flipped:", flipped)
    assert "coverage below" not in good and "explains the silhouettes better" not in good, good
    assert "view_at_plus_90 = front_image_right_side explains the silhouettes better" in flipped, flipped


def test_coverage_peaks_at_true_elevation():
    """Coverage is a usable (if shallow) signal for elevation errors: the true rig scores best."""
    alphas = _synthetic(el=10.0)
    fov = [math.radians(35.0)] * 8

    def mean_cov(el):
        c2w = mv.orbit_c2w([i * 45.0 for i in range(8)], [el] * 8, 2.2)
        return float(mv.rig_consistency(alphas, mv.relative_calc_mats(c2w), torch.tensor(fov), 1.0)[0].mean())

    true = mean_cov(10.0)
    assert true > mean_cov(-5.0) + 0.02 and true > mean_cov(25.0) + 0.02


def test_auto_fit_frames_object_to_margin():
    """auto_fit rescales the rig so the object spans at most 0.5/margin of the grid (perspective
    makes it conservative), without shrinking it much below that."""
    d_true, margin = 2.2, 1.1
    alphas = _synthetic(d=d_true, el=0.0)
    img, msk = _as_comfy(alphas)
    bundle, _, report = mv.views_from_orbit(img, msk, False, "auto", "0", 35.0, "front_image_right_side",
                                            0, "auto_fit", margin, 0.0, 1.0, align="none")
    scale = float(bundle["camera_distance"][0, 0]) / d_true   # world scale of the fitted rig
    lin = torch.linspace(-0.6, 0.6, 121)
    pts = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    extent = float(pts[_inside(pts)].abs().max()) * scale
    assert 0.85 * 0.5 / margin <= extent <= 0.5 / margin, (extent, 0.5 / margin)
    assert "coverage below" not in report, report


def test_tall_frames_with_their_own_fov_match_square_frames():
    """Cropping square renders to a tall frame (with the tall frame's narrower FOV) and letting
    pad_to_square restore it must give the same rig check as the square frames."""
    fov = 35.0
    alphas = _synthetic(fov_deg=fov)
    res, w = alphas[0].shape[0], 192
    x0 = (res - w) // 2
    tall = [a[:, x0:x0 + w] for a in alphas]
    fov_tall = math.degrees(2 * math.atan(math.tan(math.radians(fov) / 2) * w / res))
    img, msk = _as_comfy(tall)
    _, _, report = mv.views_from_orbit(img, msk, False, "auto", "10", fov_tall, "front_image_right_side",
                                       0, "manual", 1.1, 2.2, 1.0, align="none")
    assert "coverage below" not in report, report


def test_front_index_rebases_azimuths():
    alphas = _synthetic(n_views=4, el=0.0)
    img, msk = _as_comfy(alphas)
    rolled = torch.roll(img, 1, 0), torch.roll(msk, 1, 0)       # the front view is now frame 1
    b, _, report = mv.views_from_orbit(rolled[0], rolled[1], False, "auto", "0", 35.0,
                                       "front_image_right_side", 1, "manual", 1.1, 2.2, 1.0, align="none")
    ref_b, _, _ = mv.views_from_orbit(img, msk, False, "auto", "0", 35.0,
                                      "front_image_right_side", 0, "manual", 1.1, 2.2, 1.0, align="none")
    assert torch.allclose(b["transform_matrix"], ref_b["transform_matrix"], atol=1e-5)
    assert torch.equal(b["images"][512], ref_b["images"][512])
    assert "WARNING" not in report, report


def _scramble(alphas, scales, offsets, sizes):
    """Re-frame each square silhouette: rescale it, drop it at an offset into a canvas of another
    size -- what separately cropped / resized frames of one orbit look like."""
    out = []
    for a, k, (dx, dy), (h, w) in zip(alphas, scales, offsets, sizes):
        img = Image.fromarray((a * 255).astype(np.uint8)).resize(
            (round(a.shape[1] * k), round(a.shape[0] * k)), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (w, h), 0)
        canvas.paste(img, ((w - img.width) // 2 + dx, (h - img.height) // 2 + dy))
        out.append(np.asarray(canvas, np.float32) / 255.0)
    return out


def test_bbox_height_alignment_repairs_mismatched_frames():
    alphas = _synthetic(n_views=4, fov_deg=25.0, el=0.0, d=2.8)
    frames = _scramble(alphas, scales=[1.0, 0.7, 1.25, 0.85], offsets=[(0, 0), (40, -25), (-30, 20), (15, 30)],
                       sizes=[(256, 256), (300, 190), (360, 330), (240, 280)])
    # A batch needs one size: center each frame on a common canvas (no resize), which keeps
    # every frame's own scale and offset mismatch intact.
    H, W = max(f.shape[0] for f in frames), max(f.shape[1] for f in frames)
    msk_b = torch.zeros(4, H, W)
    for i, f in enumerate(frames):
        y0, x0 = (H - f.shape[0]) // 2, (W - f.shape[1]) // 2
        msk_b[i, y0:y0 + f.shape[0], x0:x0 + f.shape[1]] = torch.from_numpy(f)
    img_b = msk_b[..., None].repeat(1, 1, 1, 3) * 0.5
    results = {}
    for align in ("none", "bbox_height"):
        _, _, report = mv.views_from_orbit(img_b, msk_b, False, "auto", "0", 25.0, "front_image_right_side",
                                           0, "auto_fit", 1.1, 0.0, 1.0, align=align)
        results[align] = report
        print(f"align={align}: {report.splitlines()[0]}")
    assert "coverage below" in results["none"], results["none"]
    assert "coverage below" not in results["bbox_height"], results["bbox_height"]
    assert "explains the silhouettes better" not in results["bbox_height"], results["bbox_height"]


def test_bbox_height_alignment_keeps_consistent_views_consistent():
    frames = sorted(EXAMPLE.glob("view*.png"))
    arr = torch.from_numpy(np.stack([np.asarray(Image.open(p).convert("RGBA"), np.float32) / 255.0 for p in frames]))
    _, _, report = mv.views_from_orbit(arr, None, False, "auto", "0", 20.0, "front_image_right_side",
                                       0, "auto_fit", 1.1, 0.0, 1.0, align="bbox_height")
    print("upstream example, bbox_height:", report)
    assert "WARNING" not in report, report


def test_missing_mask_is_an_error():
    img = torch.rand(2, 64, 64, 3)
    try:
        mv.comfy_views(img)
        raise AssertionError("no-mask input accepted")
    except ValueError:
        pass


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
