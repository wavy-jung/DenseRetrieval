import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from typing import Union


def matrix_mul(query_tensor: T, ctx_tensor: T) -> T:
    return torch.matmul(query_tensor, ctx_tensor)


def cosine_sim(query_tensor: T, ctx_tensor: T):
    tensor_mul = matrix_mul(query_tensor, ctx_tensor)
    pass

def calculate_mrr(indices: torch.LongTensor, targets: torch.LongTensor) -> float:
        """
        Calculates the MRR score for the given predictions and targets
        Args:
            indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
            targets (B): torch.LongTensor. actual target indices.

        Returns:
            mrr (float): the mrr score
        """
        tmp = targets.view(-1, 1)
        targets = tmp.expand_as(indices)
        hits = (targets == indices).nonzero()
        ranks = hits[:, -1] + 1
        ranks = ranks.float()
        rranks = torch.reciprocal(ranks)
        mrr = torch.sum(rranks).data / targets.size(0)
        return mrr.item() 


def calculate_hit(indices: T, targets: torch.LongTensor) -> float:
    hit_rate_T = (indices == targets.reshape(-1, 1)).any(1).float().mean()
    return float(hit_rate_T)
    
    