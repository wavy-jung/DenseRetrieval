import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T


def matrix_mul(query_tensor: T, ctx_tensor: T) -> T:
    return torch.matmul(query_tensor, ctx_tensor)


def cosine_sim():
    pass