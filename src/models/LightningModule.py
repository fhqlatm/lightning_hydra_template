import json
import math
import numpy as np
from pathlib import Path
from functools import partial
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Metric, Accuracy, MaxMetric, MeanMetric

import lightning
from lightning import LightningModule

from transformers import get_linear_schedule_with_warmup


class LightningModule(LightningModule):
    """Example of LightningModule.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self, 
        _config: DictConfig, 
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
     
        self._config = _config
        
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute() # get current val acc
        self.val_acc_best(acc) # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        param_optimizer = self.named_parameters()
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': self._config.optimizers.weight_decay
            }, 
            {
                'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self._config.optimizers.learning_rate, 
            betas=self._config.optimizers.betas, 
            eps=self._config.optimizers.eps
        )
        
        # warmup proportion assertion
        assert self._config.optimizers.warmup_proportion <= 1.0, "warmup_proportion cannot be greater than 1"
        
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer, 
            num_training_steps=self._config.optimizers.total_steps, 
            num_warmup_steps=math.ceil(self._config.optimizers.total_steps * self._config.optimizers.warmup_proportion)
        )

        scheduler = {
            "scheduler": lr_scheduler, 
            "interval": "step",
            "frequency": 1
        }

        return (
            [optimizer],
            [scheduler],
        )
    
    
if __name__ == "__main__":
    _ = LightningModule(None, None, None)
