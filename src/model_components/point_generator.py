"""
Point generator.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchtyping import TensorType

from nerfstudio.cameras.cameras import Cameras

num_points = None


class PointGenerator(nn.Module):
    """
    torch.nn Module for generating points: convert frame coordinates to world coordinates.
    This class is the interface between the scene's cameras/camera optimizer and the point
    sampler.

    Args:
        cameras: Camera objects containing camera info.
        camera_calib: order of camera axis (xyz, yxz, ...)
    """

    def __init__(self, cameras: Cameras, camera_calib: str) -> None:
        super().__init__()
        self.cameras = cameras
        self.image_coords = nn.Parameter(
            cameras.get_image_coords(), requires_grad=False
        )
        self.camera_calib = camera_calib

    def forward(
        self,
        point_indices: TensorType["num_points", 3],
        depth: TensorType["num_points", 1],
    ) -> TensorType["num_points", 2]:
        """Index into the cameras to generate the points.

        Args:
            point_indices: Contains camera, row, and col indicies for target rays.
        """
        c = point_indices[:, 0]  # camera indices
        y = point_indices[:, 1]  # row indices
        x = point_indices[:, 2]  # col indices
        coords = self.image_coords[y, x] - 0.5

        # Project pixel coordinates to image coordinates with depth and "homogenize" them
        depth = depth.unsqueeze(-1)
        X_img = torch.hstack(
            [
                torch.mul(coords[:, [1, 0]], depth),
                depth,
                torch.ones(depth.shape, device=depth.device),
            ]
        )

        # Build a NeRF to raw poses transform
        N2R = torch.eye(4).to(self.cameras.device)
        N2R[1, 1] = -1
        N2R[2, 2] = -1
        N2R = N2R.unsqueeze(0).repeat(c.shape[0], 1, 1).to(torch.float32)

        # "Homogenize" extrinsics
        E = self.cameras.camera_to_worlds[c]
        E_ = F.pad(input=E, pad=(0, 0, 0, 1, 0, 0), mode="constant", value=0)
        E_[:, 3, 3] = 1

        # "Homogenize" intrinsics
        K = self.cameras.get_intrinsics_matrices()[c].to(E_.device)
        K_ = F.pad(input=K, pad=(0, 1, 0, 1, 0, 0), mode="constant", value=0)
        K_[:, 3, 3] = 1

        # Compute image to world coordinate transformation
        w2c = (K_ @ torch.inverse((E_ @ N2R).to(torch.float32))).to(torch.float32)
        c2w = torch.inverse(w2c)

        # Compute world coordinates
        X_world = (c2w @ X_img.unsqueeze(-1).to(c2w.dtype)).squeeze()

        if self.camera_calib == "yxz":
            X_world = X_world[..., [0, 2, 1, 3]]

        return X_world
