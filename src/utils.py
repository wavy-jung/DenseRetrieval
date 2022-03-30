import torch
import random
import numpy as np

from typing import NoReturn


def set_seed(random_seed: int=42)->NoReturn:
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)


def get_device()->str:
    return "cuda" if torch.cuda.is_available() else "cpu"
