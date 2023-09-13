"""
Semantic NeRFacto
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
from torch.nn import Parameter, CrossEntropyLoss
from nerfstudio.models.base_model import Model, ModelConfig
from torchtyping import TensorType

from src.callbacks.metrics import BlueprintMetrics
from src.fields.populate_fields import populate_blueprint
from utils.colormap import get_colormap
from utils.viz_utils import draw_blueprint, draw_confusionmatrix


num_points, num_classes = None, None
height, width = None, None
num_row_grid, num_col_grid = None, None


@dataclass
class BlueprintNerfactoModelConfig(ModelConfig):
    """Blueprint ReLU Model Config"""

    _target: Type = field(default_factory=lambda: BlueprintModel)
    field_name: str = "blueprint-nerfacto"

    num_layers: int = 2
    """Number of layers."""
    hidden_dim: int = 64
    """Number of hidden neurons."""
    base_res: int = 16
    """Base resolution of the hashmap."""
    max_res: int = 1024
    """Maximum resolution of the hashmap."""
    features_per_level: int = 2
    """Number of features per level."""
    num_levels: int = 16
    """Number of levels of the hashmap."""
    log2_hashmap_size: int = 19
    """Size of the hashmap."""
    num_semantic_classes: int = 101
    """Number of semantic classes."""
    encoding_type: str = None
    """Type of encoding to use {frequency, hashmap}."""
    n_frequencies: int = 0
    """Number of frequencies for frequency encoding."""


@dataclass
class BlueprintSirenModelConfig(ModelConfig):
    """Blueprint SIREN model Config"""

    _target: Type = field(default_factory=lambda: BlueprintModel)
    field_name: str = "blueprint-siren"

    num_layers: int = 2
    """Number of layers."""
    hidden_dim: int = 64
    """Number of hidden neurons."""
    num_semantic_classes: int = 101
    """Number of semantic classes."""


class BlueprintModel(Model):
    """Blueprint model (BluNF only)

    Args:
        config: configuration to instantiate model
    """

    config: ModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        self.colors = get_colormap(self.config.num_semantic_classes + 1)[1:]

        # Fields
        self.field = populate_blueprint(self.config)

        # Losses
        self.semantic_loss = CrossEntropyLoss()

        # Metrics
        self.blunf_metrics = BlueprintMetrics(self.config.num_semantic_classes)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

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

    def get_outputs(self, point_bundle: TensorType["num_points", 4]):
        semantic = self.field(point_bundle[:, :2])
        outputs = {"blueprint": semantic}

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        target_dict = dict(semantic=batch["semantic"].to(self.device))
        outputs["blunf_pred"] = (
            torch.nn.functional.softmax(outputs["blueprint"], dim=-1)
            .argmax(dim=-1)
            .to(target_dict["semantic"].dtype)
        )
        metrics_dict = self.blunf_metrics.get_batch_metrics(outputs, target_dict)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        semantic_target = batch["semantic"].to(self.device)
        loss_dict["blueprint_loss"] = self.semantic_loss(
            outputs["blueprint"], semantic_target
        )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        # Gather all ground-truth
        target_dict = dict(
            semantic=batch["semantic"].to(self.device),
            global_blueprint=batch["global_blueprint"].to(self.device),
            mask=batch["mask"].squeeze(),
        )
        # Get image-wise blueprint class prediction
        outputs["blunf_pred"] = (
            torch.nn.functional.softmax(outputs["blueprint"], dim=-1)
            .argmax(dim=-1)
            .to(target_dict["semantic"].dtype)
        )
        # Get global blueprint class prediction
        global_coords = target_dict["global_blueprint"][:, :, :2]
        global_confidence_target = target_dict["global_blueprint"][:, :, 3]
        target_dict["global_mask"] = (global_confidence_target > 0.5).view(-1)
        out = self.get_global_blueprint(global_coords)
        global_blueprint_out = out["blueprint_blueprint"].to(self.device)
        global_coords = out["blueprint_coordinates"].cpu()
        outputs["global_blueprint_pred"] = (
            torch.nn.functional.softmax(global_blueprint_out, dim=-1)
            .argmax(dim=-1)
            .to(int)
        )

        # Compute metrics
        blunf_metrics_dict = self.blunf_metrics.get_image_metrics(outputs, target_dict)
        global_metrics_dict = self.blunf_metrics.get_global_metrics(
            outputs, target_dict
        )
        metrics_dict = {**blunf_metrics_dict, **global_metrics_dict}

        # Draw image-wise blueprint
        mask_points_bundle = outputs["point_bundle"][target_dict["mask"]].cpu()
        colored_target = self.colors[target_dict["semantic"].to(int)]
        colored_pred = self.colors[outputs["blunf_pred"]].cpu()
        mask_colored_target = colored_target[target_dict["mask"]].cpu()
        mask_colored_pred = colored_pred[target_dict["mask"]].cpu()
        blueprint_pred = draw_blueprint(mask_points_bundle, mask_colored_pred)
        blueprint_target = draw_blueprint(mask_points_bundle, mask_colored_target)
        combined_blueprint = torch.cat([blueprint_target, blueprint_pred], dim=1)

        # Draw global blueprint
        bp_width, bp_height = self.global_coords.shape[:2]
        global_blueprint_target = (
            target_dict["global_blueprint"][:, :, 2].to(int).view(-1)
        )
        global_target = -torch.ones((bp_width, bp_height)).to(global_blueprint_target)
        global_target[
            target_dict["global_mask"].view(bp_width, bp_height)
        ] = global_blueprint_target[target_dict["global_mask"]]
        colored_global_target = (
            self.colors[global_target.view(-1)].cpu().view(bp_width, bp_height, 3)
        )
        colored_global_pred = (
            self.colors[outputs["global_blueprint_pred"]]
            .cpu()
            .view(bp_width, bp_height, 3)
        )
        # global_pred = draw_blueprint(global_coords, colored_global_pred)
        combined_global = torch.cat([colored_global_target, colored_global_pred], dim=1)

        # Draw global confusion matrix
        conf_matrix = metrics_dict.pop("conf_matrix")
        confusion_plot = draw_confusionmatrix(conf_matrix.cpu())

        images_dict = dict(
            blueprint=combined_blueprint,
            global_blueprint=combined_global,
            global_confusion_matrix=confusion_plot,
        )

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_global_blueprint(
        self, global_coordinates: TensorType["num_row_grid", "num_col_grid", 2]
    ) -> Dict[str, torch.Tensor]:
        """Infer the global bluprint of a grid of coordinates."""
        num_points_per_chunk = self.config.eval_num_rays_per_chunk

        global_coordinates = global_coordinates.view(-1, 2)
        blueprint_num_points = global_coordinates.shape[0]
        blueprint_outputs_lists = defaultdict(list)
        for i in range(0, blueprint_num_points, num_points_per_chunk):
            start_idx = i
            end_idx = i + num_points_per_chunk
            point_bundle = global_coordinates[start_idx:end_idx]
            out = self.forward(point_bundle)
            for output_name, x in out.items():
                blueprint_outputs_lists[output_name].append(x)
            blueprint_outputs_lists["coordinates"].append(point_bundle)

        outputs = {}
        for output_name, outputs_list in blueprint_outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[f"blueprint_{output_name}"] = torch.cat(outputs_list)

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
        self, camera_point_bundle: TensorType["num_points", 2]
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_point_bundle: point bundle to calculate outputs over
        """
        image_height, image_width = camera_point_bundle.shape[:2]
        num_points_per_chunk = self.config.eval_num_rays_per_chunk
        num_points = image_height * image_width
        camera_point_bundle = camera_point_bundle.view(num_points, -1)

        outputs_lists = defaultdict(list)
        for i in range(0, num_points, num_points_per_chunk):
            start_idx = i
            end_idx = i + num_points_per_chunk
            point_bundle = camera_point_bundle[start_idx:end_idx]
            out = self.forward(point_bundle)
            for output_name, x in out.items():  # type: ignore
                outputs_lists[output_name].append(x)
            outputs_lists["point_bundle"].append(point_bundle)

        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(
                image_height, image_width, -1
            )

        return outputs
