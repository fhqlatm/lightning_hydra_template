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

class LoadDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path) as f:
            self.dataset = json.load(f)
            
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    