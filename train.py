import argparse
import pandas as pd
from src.utils import set_seed
from tqdm import tqdm
from typing import NoReturn

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader

from model.dpr import DenseRetriever
from arguments import TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

from inference import compute_metrics


class Trainer(object):
    def __init__(self, 
        args: TrainingArguments,
        p_encoder: nn.Module,
        q_encoder: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        test_dataloader: DataLoader = None
    ):
        pass

    def train(self,):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        t_total = len(self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        best_mrr_scores = -1

        train_iterator = tqdm(range(int(self.args.num_train_epochs)), desc="Epoch")
        for epoch in train_iterator:    
            self.train_one_epoch(
                self.p_encoder,
                self.q_encoder,
                self.train_dataloader,
                optimizer,
                scheduler
            )

            mrr_scores = self.validation(

            )

            if best_mrr_scores <= mrr_scores:
                self.save_models(epoch)


    def save_models(self, epoch):
        self.p_encoder.save_pretrained()
        self.q_encoder.save_pretrained()


    def validation():
        pass


    def train_one_epoch(
        self,
        p_encoder,
        q_encoder,
        train_dataloader,
        optimizer,
        scheduler
    ):
        batch_size = self.args.per_device_train_batch_size
        with tqdm(self.train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:

                self.p_encoder.train()
                self.q_encoder.train()

                batch_size = len(batch[0]) # 마지막 batch에서 sample수가 다른 경우 dim. error 발생 방지하기 위해 batch_size 업데이트
                targets = torch.arange(0, batch_size*(self.num_neg+1), (self.num_neg+1)).long() # ex) num_neg=1인 경우, 정답은 [0, 2, 4 ...] 인덱스
                targets = targets.to(self.args.device)
            
                p_inputs = {
                    "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(self.args.device),
                    "attention_mask": batch[1].view(batch_size * (self.num_neg+1), -1).to(self.args.device),
                    "token_type_ids": batch[2].view(batch_size * (self.num_neg+1), -1).to(self.args.device)
                }
            
                q_inputs = {
                    "input_ids": batch[3].to(self.args.device),
                    "attention_mask": batch[4].to(self.args.device),
                    "token_type_ids": batch[5].to(self.args.device)
                }

                # (batch_size*(num_neg+1), emb_dim)
                p_outputs = self.p_encoder(**p_inputs)["pooler_output"]
                # (batch_size*, emb_dim)
                q_outputs = self.q_encoder(**q_inputs)["pooler_output"]

                # Calculate similarity score & loss
                sim_scores = torch.matmul(q_outputs, p_outputs.T).squeeze() #(batch_size, num_neg + 1)
                sim_scores = sim_scores.view(batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)
                tepoch.set_postfix(loss=f"{str(loss.item())}")

                loss.backward()
                optimizer.step()
                scheduler.step()

                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()

                global_step += 1

                torch.cuda.empty_cache()


    def inference():
        pass



def set_loader():
    pass


def doc2id():
    pass


def doc2embedding():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arguments")
    parser.add_argument("--")
    set_seed()