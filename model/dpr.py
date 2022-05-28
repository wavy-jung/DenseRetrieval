import numpy as np
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer, AutoModel,
    BertModel, BertPreTrainedModel
)
# from transformers import DPRContextEncoder, DPRQuestionEncoder


class BertEncoder(nn.Module):
    def __init__(self, checkpoint = "bert-base-uncased"):
        super(BertEncoder, self).__init__()

        self.bert = AutoModel.from_pretrained(checkpoint)
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs["pooler_output"]
        return pooled_output


class Seq2SeqEncoder(nn.Module):
    pass


if __name__=="__main__":
    p_encoder = BertEncoder()
    q_encoder = BertEncoder()