"""Data parser for blender dataset"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import os.path as osp
from pathlib import Path
from typing import Any, Dict, Type

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
import numpy as np
import torch

from utils.file_utils import load_pth


def read_semantic_traj(file, selected=None):
    l_cam = []
    with open(file) as f:
        lines = f.readlines()
    for idx in range(len(lines)):
        numbers_list = []
        for num_str in lines[idx].split():
            num_int = float(num_str)
            numbers_list.append(num_int)
        l_cam.append(np.array(numbers_list).reshape(4, 4))

    return l_cam


@dataclass
class ReplicaDataParserConfig(DataParserConfig):
    """Replica dataset parser config"""

    _target: Type = field(default_factory=lambda: Replica)
    """target class to instantiate"""
    data: Path = Path("data/replica")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    semantic: Dict[str, Any] = field(default_factory=list)
    """Semantic metadata classes informations (num classes, LUT, ...)"""
    ranking: Dict[str, Any] = field(default_factory=list)
    """Height ranking labels"""


@dataclass
class Replica(DataParser):
    """Replica Dataset"""

    config: ReplicaDataParserConfig

    def __init__(self, config: ReplicaDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.semantic_metadata: Dict[str, Any] = config.semantic
        self.ranking_metadata: Dict[str, Any] = config.ranking
        self.scale_factor: float = config.scale_factor
        self.train_split_percentage = config.train_split_percentage
        self.rescale_factor = 1

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.config.data / "transforms.json")
        traj_w_c = np.array(read_semantic_traj(self.config.data / "traj_w_c.txt"))

        # load files paths and poses
        image_filenames, semantic_filenames, depth_filenames = [], [], []
        poses, raw_poses = [], []
        for frame_index, frame in enumerate(meta["frames"]):
            image_filenames.append(self.config.data / frame["rgb_path"])
            semantic_filenames.append(self.config.data / frame["semantic_path"])
            depth_filenames.append(self.config.data / frame["depth_path"])
            poses.append(np.array(frame["transform_matrix"]))
            raw_poses.append(traj_w_c[frame_index])

        assert (
            len(image_filenames) != 0
        ), """
        No image files found. You should check the file_paths in the transforms.json file
        to make sure they are correct.
        """

        # filter image_filenames and poses based on train/eval split percentage
        num_images = len(image_filenames)
        num_train_images = math.ceil(num_images * self.train_split_percentage)
        num_eval_images = num_images - num_train_images
        i_all = np.arange(num_images)
        i_train = np.linspace(
            0, num_images - 1, num_train_images, dtype=int
        )  # equally spaced training images starting and ending at 0 and num_images-1
        i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
        assert len(i_eval) == num_eval_images
        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        # Scale poses
        scale_factor = 1.0
        scale_factor /= torch.max(torch.abs(poses[:, :3, 3])) * self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        image_filenames = [image_filenames[i] for i in indices]
        semantic_filenames = [semantic_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices]
        poses = poses[indices]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        # aabb_scale = self.config.scene_scale
        aabb_scale = 2
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = float(meta["fl_x"])
        fy = float(meta["fl_y"])
        cx = float(meta["cx"])
        cy = float(meta["cy"])
        height = int(meta["h"])
        width = int(meta["w"])
        distortion_params = camera_utils.get_distortion_params(
            k1=float(meta["k1"]) if "k1" in meta else 0.0,
            k2=float(meta["k2"]) if "k2" in meta else 0.0,
            k3=float(meta["k3"]) if "k3" in meta else 0.0,
            k4=float(meta["k4"]) if "k4" in meta else 0.0,
            p1=float(meta["p1"]) if "p1" in meta else 0.0,
            p2=float(meta["p2"]) if "p2" in meta else 0.0,
        )

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        metadata = dict(
            semantic_filenames=semantic_filenames,
            depth_filenames=depth_filenames,
            semantic_metadata=self.semantic_metadata,
            ranking_metadata=self.ranking_metadata,
            scale_factor=scale_factor,
        )
        if osp.exists(self.config.data / "global_blueprint.pth"):
            global_blueprint = load_pth(self.config.data / "global_blueprint.pth")
            metadata["global_blueprint"] = global_blueprint

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            metadata=metadata,
            cameras=cameras,
            scene_box=scene_box,
        )

        return dataparser_outputs
