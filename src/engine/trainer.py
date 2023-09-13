"""
Code to train model.
"""
from __future__ import annotations

import cv2
import dataclasses
import functools
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from rich.progress import Console, track
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter

from utils.file_utils import create_dir, save_pickle
from src.callbacks.metrics import BlueprintMetrics
from src.models.nerfacto import NerfactoModel
from src.models.stereo import StereoModel

CONSOLE = Console(width=120)
SAVE_BLUEPRINT = True

num_samples, height, width, num_channels = None, None, None, None


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision = self.config.mixed_precision
        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.mixed_precision)

        self.base_dir = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")
        # set up viewer if enabled
        self.viewer_state, banner_messages = None, None
        self._check_viewer_warnings()
        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / config.logging.relative_log_dir
        self.log_dir = writer_log_path
        writer.setup_event_writer(
            config.is_wandb_enabled(),
            config.is_tensorboard_enabled(),
            log_dir=writer_log_path,
        )
        writer.setup_local_writer(
            config.logging,
            max_iter=config.max_num_iterations,
            banner_messages=banner_messages,
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)
        profiler.setup_profiler(config.logging)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val"):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datset into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
        )
        self.optimizers = self.setup_optimizers()

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,  # type: ignore
                grad_scaler=self.grad_scaler,  # type: ignore
                pipeline=self.pipeline,  # type: ignore
            )
        )

        # TODO: to fix ugly/tricky way to store global coordinates for BluNF
        global_coords = self.pipeline.datamanager.eval_dataset.global_blueprint[
            :, :, :2
        ]
        x_min, x_max = global_coords[:, :, 0].min(), global_coords[:, :, 0].max()
        y_min, y_max = global_coords[:, :, 1].min(), global_coords[:, :, 1].max()
        num_xsteps, num_ysteps = global_coords.shape[:2]
        x_step = (x_max - x_min) / (num_xsteps - 1)
        y_step = (y_max - y_min) / (num_ysteps - 1)
        grid_offset = torch.tensor([x_min, y_min])
        grid_step = torch.tensor([x_step, y_step])

        self.pipeline.model.global_coords = global_coords.to(self.device)
        self.pipeline.model.grid_offset = grid_offset.to(self.device)
        self.pipeline.model.grid_step = grid_step.to(self.device)

        if isinstance(self.pipeline.model, StereoModel):
            num_train_samples = len(self.pipeline.datamanager.train_dataloader)
            _, point_bundles, batches = self.pipeline.datamanager.get_train_images()
            point_bundles = torch.stack(point_bundles).view(num_train_samples, -1, 4)
            semantics = torch.stack([x["semantic"] for x in batches]).view(
                num_train_samples, -1
            )
            self.pipeline.model.get_outputs(point_bundles, semantics)

    def setup_optimizers(self) -> Optimizers:
        """Helper to set up the optimizers
        Returns:
            The optimizers object given the trainer config.
        """
        optimizer_config = self.config.optimizers.copy()
        camera_optimizer_config = self.config.pipeline.datamanager.camera_optimizer
        param_groups = self.pipeline.get_param_groups()
        if camera_optimizer_config.mode != "off":
            assert camera_optimizer_config.param_group not in optimizer_config
            optimizer_config[camera_optimizer_config.param_group] = {
                "optimizer": camera_optimizer_config.optimizer,
                "scheduler": camera_optimizer_config.scheduler,
            }
        return Optimizers(optimizer_config, param_groups)

    def train(self) -> None:
        """Train the model."""
        assert (
            self.pipeline.datamanager.train_dataset is not None
        ), "Missing DatsetInputs"

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations
            for step in range(self._start_step, self._start_step + num_iterations):
                with TimeWriter(
                    writer, EventName.ITER_TRAIN_TIME, step=step
                ) as train_t:
                    self.pipeline.train()

                    # training callbacks before the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step,
                            location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION,
                        )

                    # time the forward pass
                    loss, loss_dict, metrics_dict = self.train_iteration(step)

                    # training callbacks after the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step,
                            location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION,
                        )

                # Skip the first two steps to avoid skewed timings that break the viewer
                # rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.config.pipeline.datamanager.train_num_rays_per_batch
                        / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(
                    step, self.config.logging.steps_per_log, run_at_zero=True
                ):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(
                        name="Train Loss Dict", scalar_dict=loss_dict, step=step
                    )
                    writer.put_dict(
                        name="Train Metrics Dict", scalar_dict=metrics_dict, step=step
                    )

                self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()
            # save checkpoint at the end of training
            self.save_checkpoint(step)

            CONSOLE.rule()
            CONSOLE.print(
                "[bold green]:tada: :tada: :tada: Training Finished :tada: :tada: :tada:",
                justify="center",
            )
            if not self.config.viewer.quit_on_train_completion:
                CONSOLE.print("Use ctrl+c to quit", justify="center")
                self._always_render(step)

    @torch.no_grad()
    def infer_depth(self):
        num_points_per_chunk = 2 * self.pipeline.model.config.eval_num_rays_per_chunk
        datamanager = self.pipeline.datamanager
        camera_scale_factor = datamanager.train_dataset.camera_scale_factor

        # Compute training depth maps
        num_training_samples = len(datamanager.train_dataloader)
        for step in track(
            range(num_training_samples),
            description="Inferring training samples",
            transient=True,
        ):
            count = datamanager.train_dataloader.count
            if count == num_training_samples:
                break

            ray_bundle, batch = next(datamanager.train_dataloader)
            index = ray_bundle.camera_indices[0, 0].item()
            filename = datamanager.train_dataset.depth_filenames[index]
            saving_dir = filename.parent.parent / "nerf_depth"
            create_dir(saving_dir)
            if (saving_dir / filename.name).exists():
                continue

            w, h = ray_bundle.shape
            ray_bundle = ray_bundle.reshape(-1)
            total_num_points = ray_bundle.shape[0]

            batch_depth = []
            for i in range(0, total_num_points, num_points_per_chunk):
                start_idx = i
                end_idx = i + num_points_per_chunk
                out = self.pipeline.model(ray_bundle[start_idx:end_idx])
                batch_depth.append(out["depth"].cpu())

            depth = torch.cat(batch_depth).reshape(w, h)
            depth /= camera_scale_factor
            depth *= 1000
            cv2.imwrite(
                str(saving_dir / filename.name), depth.numpy().astype(np.uint16)
            )

        # Compute eval depth maps
        num_eval_samples = len(datamanager.fixed_indices_eval_dataloader)
        for _ in track(
            range(len(datamanager.fixed_indices_eval_dataloader)),
            description="Inferring eval samples",
            transient=True,
        ):
            count = datamanager.fixed_indices_eval_dataloader.count
            if count == num_eval_samples:
                break

            ray_bundle, batch = next(datamanager.fixed_indices_eval_dataloader)
            index = ray_bundle.camera_indices[0, 0].item()
            filename = datamanager.eval_dataset.depth_filenames[index]
            saving_dir = filename.parent.parent / "nerf_depth"
            if (saving_dir / filename.name).exists():
                continue
            w, h = ray_bundle.shape
            ray_bundle = ray_bundle.reshape(-1)
            total_num_points = ray_bundle.shape[0]

            batch_depth = []
            for i in range(0, total_num_points, num_points_per_chunk):
                start_idx = i
                end_idx = i + num_points_per_chunk
                out = self.pipeline.model(ray_bundle[start_idx:end_idx])
                batch_depth.append(out["depth"].cpu())

            depth = torch.cat(batch_depth).reshape(w, h)
            depth /= camera_scale_factor
            depth *= 1000
            cv2.imwrite(
                str(saving_dir / filename.name), depth.numpy().astype(np.uint16)
            )

        CONSOLE.print(f"Depth maps saved at: {saving_dir}")

    @torch.no_grad()
    def infer_eval(self):
        """Infer the model over the eval dataset."""
        model = self.pipeline.model.eval()
        datamanager = self.pipeline.datamanager

        # Warning to the data config for the number of classes
        num_blueprint_classes = len(
            datamanager.config.dataparser.semantic.class_to_keep
        )
        num_metric_class = num_blueprint_classes
        # Add unseen classes for 'StereoModel' and 'NerfactoModel'
        if isinstance(model, StereoModel) or isinstance(model, NerfactoModel):
            num_metric_class += 1
        blunf_metrics = BlueprintMetrics(num_metric_class).to(self.device)

        # Get target data
        target = datamanager.eval_dataset.global_blueprint.contiguous()
        coords = datamanager.eval_dataset.global_blueprint[:, :, :2]
        confidence = datamanager.eval_dataset.global_blueprint[:, :, 3]
        mask = (confidence > 0.5).view(-1)

        out = model.get_global_blueprint(coords.contiguous().to(model.device))
        out = out["blueprint_blueprint"].nan_to_num()
        pred = torch.nn.functional.softmax(out, dim=-1).argmax(dim=-1).to(int)
        pred[pred >= num_blueprint_classes] = num_blueprint_classes
        print(pred.max())
        target_dict = dict(
            global_mask=mask.to(self.device), global_blueprint=target.to(self.device)
        )
        pred_dict = dict(global_blueprint_pred=pred.to(self.device))
        eval_outputs = blunf_metrics.get_global_metrics(pred_dict, target_dict)
        eval_outputs["pred"] = pred
        eval_outputs["target"] = target
        comp = 1 - (pred == (target.max() + 1)).sum() / pred.shape[0]
        eval_outputs["blueprint/global/comp"] = comp
        eval_outputs = {k: v.cpu() for k, v in eval_outputs.items()}
        save_path = self.base_dir / "eval_outputs.pkl"
        save_pickle(eval_outputs, save_path)

        pACC = eval_outputs["blueprint/global/pACC"]
        fwIoU = eval_outputs["blueprint/global/fwIoU"]
        CONSOLE.print(
            f"pACC: {pACC:.2%} / " f"fwIoU: {fwIoU:.2%} / " f"Completeness: {comp:.2%}"
        )

        CONSOLE.print(f"Outputs saved at {save_path}")

    @check_main_thread
    def _always_render(self, step):
        if self.config.is_viewer_enabled():
            while True:
                self.viewer_state.vis["renderingState/isTraining"].write(False)
                self._update_viewer_state(step)

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """
        Helper to print out any warnings regarding the way the viewer/loggers are enabled
        """
        if self.config.is_viewer_enabled():
            string = (
                "[NOTE] Not running eval iterations since only viewer is enabled."
                " Use [yellow]--vis wandb[/yellow] or [yellow]--vis tensorboard[/yellow]"
                " to run with eval instead."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            dataset=self.pipeline.datamanager.train_dataset,
            start_train=self.config.viewer.start_train,
        )
        if not self.config.viewer.start_train:
            self._always_render(self._start_step)

    @check_viewer_enabled
    def _update_viewer_state(self, step: int):
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.
        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        with TimeWriter(writer, EventName.ITER_VIS_TIME, step=step) as _:
            num_rays_per_batch = (
                self.config.pipeline.datamanager.train_num_rays_per_batch
            )
            try:
                self.viewer_state.update_scene(
                    self, step, self.pipeline.model, num_rays_per_batch
                )
            except RuntimeError:
                time.sleep(0.03)  # sleep to allow buffer to reset
                assert self.viewer_state.vis is not None
                self.viewer_state.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from"
                    " crashing."
                )

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(
        self, train_t: TimeWriter, vis_t: TimeWriter, step: int
    ):
        """Performs update on rays/sec calclation for training
        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch = (
            self.config.pipeline.datamanager.train_num_rays_per_batch
        )
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        if load_dir is not None:
            # Check if `load_dir` correspond to a pre-computed blueprint path
            if load_dir.suffix == ".pth":
                self.precomputed_blueprint = torch.load(load_dir)
                CONSOLE.print("No checkpoints to load, training from scratch")
                return
            # Check if `load_dir` correspond to the checkpoint path
            elif load_dir.suffix == ".ckpt":
                load_path = load_dir
            # Otherwise, load the latest checkpoint from `load_dir`
            else:
                load_step = self.config.load_step
                if load_step is None:
                    print("Loading latest checkpoint from load_dir")
                    # NOTE: this is specific to the checkpoint name format
                    load_step = sorted(
                        int(x[x.find("-") + 1 : x.find(".")])
                        for x in os.listdir(load_dir)
                    )[-1]
                load_path = load_dir / f"step-{load_step:09d}.ckpt"

            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {
                    k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()
                },
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    @profiler.time_function
    def train_iteration(
        self, step: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        self.optimizers.zero_grad_all()
        cpu_or_cuda_str = self.device.split(":")[0]
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
        self.grad_scaler.update()
        self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step):
        """
        Run one iteration with different batch/image/all image evaluations depending on
        step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch, run_at_zero=True):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(
                step=step
            )
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(
                name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step
            )
            writer.put_dict(
                name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step
            )

        # one eval image
        if step_check(step, self.config.steps_per_eval_image, run_at_zero=True):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                (
                    metrics_dict,
                    images_dict,
                ) = self.pipeline.get_eval_image_metrics_and_images(step=step)

            if "_global_blueprint" in images_dict and SAVE_BLUEPRINT:
                pred_blueprint_dir = self.log_dir / "pred_blueprint"
                create_dir(pred_blueprint_dir)
                global_blueprint = images_dict.pop("_global_blueprint")
                global_coords = images_dict.pop("_global_coords")
                global_data = torch.hstack([global_blueprint[..., None], global_coords])
                torch.save(global_data, pred_blueprint_dir / f"step_{step:05}.pth")

            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(
                name="Eval Images Metrics", scalar_dict=metrics_dict, step=step
            )
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(
                name="Eval Images Metrics Dict (all images)",
                scalar_dict=metrics_dict,
                step=step,
            )
