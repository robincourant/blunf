"""
Datamanager w/o ray, only points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from rich.progress import Console, track
from torch.nn import Parameter
from typing_extensions import Literal
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.pixel_samplers import EquirectangularPixelSampler, PixelSampler
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import DataManager

from src.data.datasets.rgbsem_dataset import RgbSemDataset
from src.model_components.point_generator import PointGenerator

CONSOLE = Console(width=120)


@dataclass
class BlueprintDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping
    the train/eval dataparsers; After instantiation, data manager holds both train/eval
    datasets and is in charge of returning unpacked train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: BlueprintDataManager)
    """Target class to instantiate."""
    dataparser: DataParserConfig = BlenderDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 1024
    """Number of points per batch to use per training iteration."""
    train_num_images_to_sample_from: int = -1
    """Number of images to sample during training iteration."""
    train_num_times_to_repeat_images: int = -1
    """When not training on all images, number of iterations before picking new
    images. If -1, never pick new images."""
    eval_num_points_per_batch: int = 1024
    """Number of points per batch to use per eval iteration."""
    eval_num_images_to_sample_from: int = -1
    """Number of images to sample during eval iteration."""
    eval_num_times_to_repeat_images: int = -1
    """When not evaluating on all images, number of iterations before picking
    new images. If -1, never pick new images."""
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    """Specifies the image indices to use during eval; if None, uses all."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """Specifies the camera pose optimizer used during training. Helpful if poses are
    noisy, such as for data from Record3D."""
    collate_fn = staticmethod(nerfstudio_collate)
    """Specifies the collate function to use for the train and eval dataloaders."""
    camera_res_scale_factor: float = 1.0
    """The scale factor for scaling spatial data such as images, mask, semantics
    along with relevant information about camera intrinsics
    """
    camera_calib: str = "xyz"
    """Order of camera axis."""
    semantic_offset: int = 0
    """Offset on semantic indexing (1 for mp3d)"""


class BlueprintDataManager(DataManager):
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little
    jank under the hood. We may clean this up a little bit under the hood with more
    standard dataloading components that can be strung together, but it can be just used
    as a black box for now since only the constructor is likely to change in the future,
    or maybe passing in step number to the next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: BlueprintDataManagerConfig
    train_dataset: RgbSemDataset
    eval_dataset: RgbSemDataset

    def __init__(
        self,
        config: BlueprintDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser = self.config.dataparser.setup()

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        super().__init__()

    def create_train_dataset(self) -> RgbSemDataset:
        """Sets up the data loaders for training"""
        return RgbSemDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split="train"),
            scale_factor=self.config.camera_res_scale_factor,
            semantic_offset=self.config.semantic_offset,
        )

    def create_eval_dataset(self) -> RgbSemDataset:
        """Sets up the data loaders for evaluation"""
        return RgbSemDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(
                split=self.test_split
            ),
            scale_factor=self.config.camera_res_scale_factor,
            semantic_offset=self.config.semantic_offset,
        )

    def _get_pixel_sampler(
        self, dataset: RgbSemDataset, *args: Any, **kwargs: Any
    ) -> PixelSampler:
        """Infer pixel sampler to use."""
        # If all images are equirectangular, use equirectangular pixel sampler
        is_equirectangular = (
            dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value
        )
        if is_equirectangular.all():
            return EquirectangularPixelSampler(*args, **kwargs)
        # Otherwise, use the default pixel sampler
        if is_equirectangular.any():
            CONSOLE.print(
                "[bold yellow]Warning: Some cameras are equirectangular, but using "
                "default pixel sampler."
            )
        return PixelSampler(*args, **kwargs)

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.train_pixel_sampler = self._get_pixel_sampler(
            self.train_dataset, self.config.train_num_rays_per_batch
        )
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_point_generator = PointGenerator(
            self.train_dataset.cameras.to(self.device), self.config.camera_calib
        )

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(
            self.eval_dataset, self.config.eval_num_points_per_batch
        )
        self.eval_point_generator = PointGenerator(
            self.eval_dataset.cameras.to(self.device), self.config.camera_calib
        )
        # for loading full images
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            image_indices=self.config.eval_image_indices,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        point_indices = batch["indices"]
        point_bundle = self.train_point_generator(
            point_indices, batch["depth"].squeeze()
        )
        return point_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        batch = self.eval_pixel_sampler.sample(image_batch)
        point_indices = batch["indices"]
        point_bundle = self.eval_point_generator(
            point_indices, batch["depth"].squeeze()
        )
        return point_bundle, batch

    def get_train_images(self) -> List[Tuple[int, RayBundle, Dict]]:
        image_indices, point_bundles, image_batches = [], [], []
        for camera_ray_bundle, image_batch in track(
            self.train_dataloader,
            description="Collecting data batch",
            transient=True,
        ):
            # Get image
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])

            # Get blueprint
            image_batch["global_blueprint"] = self.train_dataset.global_blueprint
            height, width = image_batch["semantic"].shape
            coords = torch.cartesian_prod(torch.arange(height), torch.arange(width))
            point_indices = torch.hstack(
                [image_idx * torch.ones([coords.shape[0], 1]), coords]
            ).long()
            point_bundle = self.train_point_generator(
                point_indices, image_batch["depth"].view(-1)
            ).view(height, width, -1)

            image_indices.append(image_idx)
            point_bundles.append(point_bundle)
            image_batches.append(image_batch)

        return image_indices, point_bundles, image_batches

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        for camera_ray_bundle, image_batch in self.eval_dataloader:
            # Get image
            assert camera_ray_bundle.camera_indices is not None
            image_idx = int(camera_ray_bundle.camera_indices[0, 0, 0])

            # Get blueprint
            image_batch["global_blueprint"] = self.eval_dataset.global_blueprint
            height, width = image_batch["semantic"].shape
            coords = torch.cartesian_prod(torch.arange(height), torch.arange(width))
            point_indices = torch.hstack(
                [image_idx * torch.ones([coords.shape[0], 1]), coords]
            ).long()
            point_bundle = self.eval_point_generator(
                point_indices, image_batch["depth"].view(-1)
            ).view(height, width, -1)
            return image_idx, point_bundle, image_batch
        raise ValueError("No more eval images")

    def get_param_groups(
        self,
    ) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.train_camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups
