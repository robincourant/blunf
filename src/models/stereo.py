"""
Semantic NeRFacto
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Type

import pandas as pd
from rich.progress import track
import torch
from torch.nn import Parameter
from nerfstudio.models.base_model import Model, ModelConfig
from torchtyping import TensorType

from src.callbacks.metrics import BlueprintMetrics
from utils.colormap import get_colormap


num_batch, num_points, num_classes = None, None, None
height, width = None, None
num_row_grid, num_col_grid = None, None


@dataclass
class StereoModelConfig(ModelConfig):
    """Stereo Model Config"""

    _target: Type = field(default_factory=lambda: StereoModel)
    field_name: str = "blueprint-nerfacto"
    num_semantic_classes: int = 101
    """Number of semantic classes."""


class StereoModel(Model):
    """Stereo model

    Args:
        config: configuration to instantiate model
    """

    config: ModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        self.config.num_semantic_classes += 1
        super().populate_modules()
        self.colors = get_colormap(self.config.num_semantic_classes + 2)[1:]
        self.blunf_metrics = BlueprintMetrics(self.config.num_semantic_classes)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}

    def forward(
        self, point_bundle: TensorType["num_points", 4]
    ) -> Dict[str, torch.Tensor]:
        """
        Run forward starting with a ray bundle. This outputs different things depending on
        the configuration of the model and whether or not the batch is provided (whether
        or not we are training basically)

        Args:
            point_bundle: containing all the information needed to render that ray latents
            included
        """
        return self.get_outputs(point_bundle[:, :2])

    def get_outputs(
        self,
        point_bundle: TensorType["num_batch", "num_points", 4],
        semantics: TensorType["num_batch", "num_points", 1],
    ):
        # ## Match projections and 2d grid  ##
        num_xsteps, num_ysteps = self.global_coords.shape[:2]
        x_step = self.global_coords[1, 0, 0] - self.global_coords[0, 0, 0]
        y_step = self.global_coords[0, 1, 1] - self.global_coords[0, 0, 1]
        x_offset = self.global_coords[..., 0].min()
        y_offset = self.global_coords[..., 1].min()
        grid_offset = torch.tensor([x_offset, y_offset], device=self.device)
        grid_step = torch.tensor([x_step, y_step], device=self.device)

        points_w = point_bundle[:, :, [0, 1]]
        indices = torch.round((points_w - grid_offset) / grid_step)
        # Mask frame-projected coordinates cropped in blueprint
        # mask_indices = torch.logical_and(indices[..., 0] < 460, indices[..., 1] < 260)
        mask_indices = torch.logical_and(
            indices[..., 0] < num_xsteps, indices[..., 1] < num_ysteps
        )
        flat_indices = indices[:, :, 0] * num_ysteps + indices[:, :, 1]

        # ## Build the ground-truth blueprint ##
        num_frames = points_w.shape[0]
        num_bp_points = num_xsteps * num_ysteps
        blueprint_target = -2 * torch.ones((num_frames, num_bp_points)).to(int)
        for frame_index in track(
            range(num_frames), description="Building stereo map", transient=True
        ):
            # for frame_index in range(num_frames):
            frame_df = pd.DataFrame(
                {
                    "ind": flat_indices[frame_index][mask_indices[frame_index]].cpu(),
                    "sem": semantics[frame_index][mask_indices[frame_index]].cpu(),
                }
            )
            frame_df = frame_df[frame_df["sem"] != -1]
            frame_df["count"] = frame_df.groupby(["ind", "sem"])["sem"].transform(
                "count"
            )
            frame_df = frame_df.drop_duplicates()
            frame_df = frame_df.sort_values(
                by=["ind", "count"], ascending=False
            ).drop_duplicates(subset=["ind"])

            frame_ind = torch.from_numpy(frame_df["ind"].to_numpy()).to(int)
            frame_sem = torch.from_numpy(frame_df["sem"].to_numpy()).to(int)
            blueprint_target[frame_index, frame_ind] = frame_sem

        # ## Compute final blueprint (w/ label max pooling over frames) and label conf
        num_coords = blueprint_target.shape[1]
        clean_target = -1 * torch.ones(num_coords)
        confidence_target = torch.zeros(num_coords)
        for coord_index in track(
            range(num_coords), description="Cleaning stereo map", transient=True
        ):
            mask = blueprint_target[:, coord_index] != -2
            if True not in mask:
                continue
            values = blueprint_target[:, coord_index][mask].numpy()
            counts = Counter(values)
            max_sem = max(counts, key=counts.get)
            confidence = counts[max_sem] / values.size
            clean_target[coord_index] = max_sem
            confidence_target[coord_index] = confidence

        # ## Save the coordinates (0, 1), ground-truth blueprint label (2) and conf (3)
        global_blueprint = torch.hstack(
            [
                self.global_coords.reshape(-1, 2).to(self.device),
                clean_target.unsqueeze(-1).to(self.device),
                confidence_target.unsqueeze(-1).to(self.device),
            ]
        )
        self.global_blueprint = global_blueprint.reshape(num_xsteps, num_ysteps, 4)

    @torch.no_grad()
    def get_global_blueprint(
        self, global_coordinates: TensorType["num_row_grid", "num_col_grid", 2]
    ) -> Dict[str, torch.Tensor]:
        """Infer the global bluprint of a grid of coordinates."""
        out = self.global_blueprint[:, :, 2]
        out[out == -1] = self.config.num_semantic_classes - 1
        one_hot = torch.eye(self.config.num_semantic_classes)[out.reshape(-1).long()]
        outputs = {"blueprint_blueprint": one_hot.to(self.device)}

        return outputs
