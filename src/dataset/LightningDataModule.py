import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from lightning import LightningDataModule

from transformers import AutoTokenizer


class LihgningDataModule(LightningDataModule):
    """LightningDataModule for dataset.

    ```A DataModule implements 6 key methods:
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self, 
        _config: DictConfig, 
    ):
        super().__init__()
        self._config = _config
        self.tokenizer = AutoTokenizer.from_pretrained(self._config.tokenizer.model_name_or_path)

    # def prepare_data(self):
    #     NotImplemented
        
    def setup(self, stage: str = None):        
        if stage == "fit" or stage is None:
            # train dataset
            with open(self._config["train_data_path"], 'r') as f:
                self.train_dataset = json.load(f)
            
            # dev dataset
            with open(self._config["dev_data_path"], 'r') as f:
                self.dev_dataset = json.load(f)

            # train, dev sampler
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.dev_sampler = DistributedSampler(self.dev_dataset, shuffle=False)
            # self.dev_sampler = SequentialSampler(self.dev_dataset)
        
        elif stage == "test" or stage is None:
            # test dataset
            with open(self._config["test_data_path"], 'r') as f:
                self.test_dataset = json.load(f)
            
            # test sampler
            self.test_sampler = SequentialSampler(self.test_dataset)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self._config.per_gpu_train_batch_size,
            sampler=self.train_sampler,
            num_workers=self._config.num_workers,
            collate_fn=self.custom_collate_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self._config.per_gpu_eval_batch_size,
            sampler=self.val_sampler,
            num_workers=self._config.num_workers,
            collate_fn=self.custom_collate_fn,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self._config.per_gpu_eval_batch_size,
            sampler=self.test_sampler,
            num_workers=self._config.num_workers,
            collate_fn=self.custom_collate_fn,
        )
        return loader
    
    # collate function (tokenize batch or etc...)
    def custom_collate_fn(self, batch: Any):
        batch = list(filter(lambda x: x is not None, batch))
        
        if len(batch) == 0:
            return None
        
        batch = self.tokenizer(
            batch,
            max_length=self._config.tokenizer.max_seq_len,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        return batch
    
    