from pathlib import Path
from typing import Dict

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.base_dataparser import DataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from omegaconf import DictConfig

from src.data.dataparsers.replica_dataparser import ReplicaDataParserConfig
from src.data.datamanagers.semantic_datamanager import SemanticDataManagerConfig
from src.data.datamanagers.blueprint_datamanager import BlueprintDataManagerConfig


def _initialize_dataparser(cfg: Dict) -> DataParserConfig:
    parser_name = cfg.pop("name")
    if parser_name == "blender":
        dataparser = BlenderDataParserConfig(**cfg)
    elif parser_name == "replica":
        dataparser = ReplicaDataParserConfig(**cfg)
    elif parser_name == "mp3d":
        dataparser = ReplicaDataParserConfig(**cfg)
    else:
        raise NotImplementedError(f"Data parser: {parser_name} is not implemented!")

    return dataparser


def _initialize_camera(cfg: Dict) -> CameraOptimizerConfig:
    cfg["optimizer"] = AdamOptimizerConfig(
        lr=cfg.pop("optimizer_lr"),
        eps=cfg.pop("optimizer_eps"),
        weight_decay=cfg.pop("optimizer_wd"),
    )
    cfg["scheduler"] = SchedulerConfig(max_steps=cfg.pop("scheduler_steps"))
    camera = CameraOptimizerConfig(**cfg)
    return camera


def initialize_datamanager(cfg: DictConfig) -> VanillaDataManagerConfig:
    cfg.data_dir = Path(cfg.data_dir)
    cfg.dataparser.data = cfg.data_dir
    dataparser = _initialize_dataparser(dict(cfg.dataparser))
    camera = _initialize_camera(dict(cfg.camera))
    cfg = dict(cfg)
    cfg.pop("data_dir")
    cfg.pop("camera")
    cfg["dataparser"] = dataparser
    cfg["camera_optimizer"] = camera

    manager_name = cfg.pop("name")
    if isinstance(dataparser, ReplicaDataParserConfig):
        if manager_name == "blueprint":
            datamanager = BlueprintDataManagerConfig(**cfg)
        else:
            datamanager = SemanticDataManagerConfig(**cfg)
    else:
        datamanager = VanillaDataManagerConfig(**cfg)
    return datamanager
