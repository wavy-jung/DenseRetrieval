import numpy as np
import tqdm
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer, AutoModel,
    BertModel, BertPreTrainedModel,
    T5Tokenizer, T5ForConditionalGeneration
)
# from transformers import DPRContextEncoder, DPRQuestionEncoder

# TODO: add bi-encoder architecture vs. single-encoder architecture with dual-training...?


class BertEncoder(nn.Module):
    def __init__(self, checkpoint = "bert-base-uncased"):
        super(BertEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(checkpoint)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None) -> T: 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs["pooler_output"]
        return pooled_output


class BiEncoder(nn.Module):
    def __init__(self, q_encoder: nn.Module, p_encoder: nn.Module):
        super(BiEncoder, self).__init__()
        assert q_encoder and p_encoder, "q_encoder and p_encoder must be inserted"
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder

    def forward(self, q_inputs, p_inputs) -> Tuple[T, T]:
        q_output: T = self.get_pooler_representation(self.q_encoder, q_inputs)
        p_output: T = self.get_pooler_representation(self.p_encoder, p_inputs)
        return q_output, p_output
    
    @torch.no_grad()
    def get_relevant_docs(self, query, documents):
        raise NotImplementedError

    def get_pooler_representation(self, model: nn.Module, model_inputs):
        return model(**model_inputs)



class Seq2SeqEncoder(nn.Module):
    # TODO: DSI structure
    pass


if __name__=="__main__":
    p_encoder = BertEncoder()
    q_encoder = BertEncoder()
    biencoder = BiEncoder(q_encoder, p_encoder)