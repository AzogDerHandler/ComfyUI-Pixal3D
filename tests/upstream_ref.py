"""Verbatim upstream reference functions (TencentARC/Pixal3D f7cf384) for tests/test_mv_views.py.

Extracted from pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py and
inference_mv.py so the tests can compare against upstream without importing the
pixal3d package (which needs CUDA extensions)."""


import numpy as np
import torch
from PIL import Image
from typing import *


def project_points_to_image_batch(
    points_3d: torch.Tensor, 
    transform_matrix: torch.Tensor, 
    camera_angle_x: torch.Tensor, 
    resolution: int = 518
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D image coordinates (batch processing).
    
    Args:
        points_3d: torch.Tensor, shape [N, 3] or [B, N, 3], 3D point coordinates (in [-1, 1] range)
        transform_matrix: torch.Tensor, shape [B, 4, 4], camera transformation matrix
        camera_angle_x: torch.Tensor, shape [B], horizontal field of view angle (radians)
        resolution: int, image resolution, default 518
    
    Returns:
        points_2d: torch.Tensor, shape [B, N, 2], image coordinates [x, y]
        depth: torch.Tensor, shape [B, N], depth values
        valid_mask: torch.Tensor, shape [B, N], mask for points within view
    """
    device = points_3d.device
    B = transform_matrix.shape[0]
    
    # Ensure inputs are torch.Tensor on correct device
    if not isinstance(transform_matrix, torch.Tensor):
        transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32, device=device)
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32, device=device)
    if not isinstance(camera_angle_x, torch.Tensor):
        camera_angle_x = torch.tensor(camera_angle_x, dtype=torch.float32, device=device)
    
    # Expand points_3d to batch dimension: [N, 3] -> [B, N, 3]
    if points_3d.dim() == 2:
        points_3d_batch = points_3d.unsqueeze(0).expand(B, -1, -1)
    else:
        points_3d_batch = points_3d
    N = points_3d_batch.shape[1]
    
    # Add homogeneous coordinates: [B, N, 3] -> [B, N, 4]
    ones = torch.ones(B, N, 1, device=device, dtype=points_3d_batch.dtype)
    points_homogeneous = torch.cat([points_3d_batch, ones], dim=-1)  # [B, N, 4]
    
    # Compute world to camera transformation matrix
    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(transform_matrix.dtype)  # linalg.inv requires fp32+
    
    # Batch transform to camera coordinate system: [B, N, 4] @ [B, 4, 4]^T -> [B, N, 3]
    points_camera = torch.bmm(points_homogeneous, world_to_camera.transpose(-2, -1))[..., :3]  # [B, N, 3]
    
    # Extract camera coordinates
    x_cam = points_camera[..., 0]  # [B, N]
    y_cam = points_camera[..., 1]  # [B, N]
    z_cam = points_camera[..., 2]  # [B, N]
    
    # Depth value (Z value in camera coordinate system, note Blender camera faces -Z direction)
    depth = -z_cam  # [B, N]
    
    # Compute camera intrinsics (batch processing)
    sensor_width = 32.0  # mm
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)  # [B]
    focal_length_pixels = focal_length * resolution / sensor_width  # [B]
    
    # Expand focal_length_pixels dimension for broadcasting: [B] -> [B, 1]
    focal_length_pixels = focal_length_pixels.unsqueeze(1)  # [B, 1]
    
    # Perspective projection to NDC coordinates
    x_ndc = focal_length_pixels * x_cam / (-z_cam + 1e-8)  # [B, N]
    y_ndc = focal_length_pixels * y_cam / (-z_cam + 1e-8)  # [B, N]
    
    # Convert to image coordinates (pixel coordinates)
    x_pixel = x_ndc + resolution / 2.0  # [B, N]
    y_pixel = -y_ndc + resolution / 2.0  # [B, N], flip Y axis
    
    # Create validity mask (points within image range and in front of camera)
    valid_mask = (
        (x_pixel >= 0) & (x_pixel < resolution) &
        (y_pixel >= 0) & (y_pixel < resolution) &
        (depth > 0)  # In front of camera
    )  # [B, N]
    
    points_2d = torch.stack([x_pixel, y_pixel], dim=-1)  # [B, N, 2]
    
    return points_2d, depth, valid_mask


def compute_relative_calc_mat(
    transform_matrix: torch.Tensor,
    distance: torch.Tensor,
    front_view_transform_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the per-view projection matrix (calc_mat) that maps each view into
    the coordinate frame where the MAIN view (index 0) is snapped to the fixed
    front-view pose F (with F's camera distance set to the main view's distance).

    calc_mat_i = F @ inv(C_0) @ C_i

    where C_i = transform_matrix[:, i] (c2w). For i == 0, calc_mat_0 == F exactly,
    which reproduces the single-view behavior.

    Args:
        transform_matrix: [B, V, 4, 4] c2w matrices (index 0 = main view).
        distance: [B, V] camera distances (only index 0 used for F).
        front_view_transform_matrix: [4, 4] the canonical front-view c2w.

    Returns:
        calc_mat: [B, V, 4, 4]
    """
    B, V = transform_matrix.shape[:2]
    device = transform_matrix.device

    # Fixed front matrix with main-view distance in the translation slot.
    F = front_view_transform_matrix.to(device).unsqueeze(0).expand(B, -1, -1).clone()  # [B,4,4]
    F[:, 1, 3] = -distance[:, 0]
    F = F.unsqueeze(1)  # [B,1,4,4]

    C0 = transform_matrix[:, 0:1]  # [B,1,4,4]

    # Do the matrix math in fp32 for numerical stability (inv is sensitive).
    with torch.amp.autocast('cuda', enabled=False):
        C0f = C0.float().expand(B, V, 4, 4).reshape(B * V, 4, 4)
        Cif = transform_matrix.float().reshape(B * V, 4, 4)
        Ff = F.float().expand(B, V, 4, 4).reshape(B * V, 4, 4)
        rel = torch.bmm(torch.linalg.inv(C0f), Cif)   # inv(C_0) @ C_i
        calc = torch.bmm(Ff, rel)                      # F @ rel
    calc_mat = calc.reshape(B, V, 4, 4)
    return calc_mat


def to_cond_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """
    Turn an RGBA view into a conditioning tensor the way training read its views:
    LANCZOS resize, then premultiply by alpha so the background is black.
    """
    image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    alpha = torch.tensor(np.array(image.getchannel(3))).float() / 255.0
    rgb = torch.tensor(np.array(image.convert('RGB'))).permute(2, 0, 1).float() / 255.0
    return rgb * alpha.unsqueeze(0)
