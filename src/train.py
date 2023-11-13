import os
import json
import hydra
from pathlib import Path
from typing import List, Optional, Tuple

import lightning
import pyrootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from src.models.LightningModule import LightningModule
from src.dataset.LightningDataModule import LightningDataModule

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(_config: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        _config (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    
    # set seed for random number generators in pytorch, numpy and python.random
    if _config.get("seed"):
        lightning.seed_everything(_config.seed, workers=True)

    log.info(f"Instantiating datamodule...")
    datamodule: LightningDataModule = LightningDataModule(_config=_config.data)

    log.info(f"Instantiating model...")
    model: LightningModule = LightningModule(_config=_config.model)

    log.info("Instantiating callbacks...")
    checkpoint_callback = ModelCheckpoint(**_config.callbacks.model_checkpoint)
    lr_monitor = LearningRateMonitor()
    callbacks: List[Callback] = [checkpoint_callback, lr_monitor, ]

    log.info("Instantiating loggers...")
    tensorboard_logger = TensorBoardLogger(**_config.logger.tensorboard)
    logger: List[Logger] = [tensorboard_logger, ]

    log.info(f"Instantiating trainer...")
    trainer: Trainer = Trainer(**_config.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "_config": _config,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if _config.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if _config.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, 
            datamodule=datamodule, 
            ckpt_path=_config.ckpt_path
        )

    train_metrics = trainer.callback_metrics

    return train_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(_config: DictConfig):
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = _config["device_ids"]

    # apply extra utilities
    # (e.g. ask for tags if none are provided in _config, print config tree, etc.)
    utils.extras(_config)

    # train the model
    metric_dict, _ = train(_config)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=_config.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
