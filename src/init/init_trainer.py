from pathlib import Path
import random

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
import numpy as np
from omegaconf import DictConfig
import torch

from src.init.init_datamanager import initialize_datamanager
from src.init.init_model import initialize_model
from src.init.init_optimizer import initialize_optim
from src.engine.trainer import Trainer


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def initialize_trainer(cfg: DictConfig) -> Trainer:
    datamanager = initialize_datamanager(cfg=cfg.datamanager)
    model = initialize_model(cfg=cfg.model)
    optimizers = initialize_optim(cfg=cfg.optim)

    cfg.trainer.load_dir = Path(cfg.trainer.load_dir) if cfg.trainer.load_dir else None
    config = TrainerConfig(
        method_name=cfg.method_name,
        output_dir=Path(cfg.output_dir),
        timestamp=cfg.timestamp,
        experiment_name=cfg.experiment_name,
        vis=cfg.vis,
        pipeline=VanillaPipelineConfig(datamanager=datamanager, model=model),
        optimizers=optimizers,
        **cfg.trainer
    )
    _set_random_seed(config.machine.seed)
    trainer = Trainer(config, cfg.compnode.local_rank, cfg.compnode.world_size)
    return trainer
