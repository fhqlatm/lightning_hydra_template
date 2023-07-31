from lightning.pytorch.utilities import rank_zero_only

from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Controls which config parts are saved by lightning loggers.

    Additionally saves:
    - Number of model parameters
    """

    hparams = {}

    _config = object_dict["_config"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = _config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = _config["data"]
    hparams["trainer"] = _config["trainer"]

    hparams["callbacks"] = _config.get("callbacks")
    hparams["extras"] = _config.get("extras")

    hparams["task_name"] = _config.get("task_name")
    hparams["tags"] = _config.get("tags")
    hparams["ckpt_path"] = _config.get("ckpt_path")
    hparams["seed"] = _config.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
