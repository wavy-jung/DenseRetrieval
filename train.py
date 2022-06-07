import os
import argparse
import pandas as pd
from utils.seed_utils import set_seed
from tqdm import tqdm
from typing import Dict, List

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from model.dpr import BertEncoder
from arguments import TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup

from test import compute_metrics

# TODO
# Add GradCache training ver.


class DualTrainer(object):
    def __init__(self, 
        args: TrainingArguments,
        p_encoder: nn.Module,
        q_encoder: nn.Module,
        passage_with_idx: Dict[int, str],
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        doc_dataloader: DataLoader,
        test_dataloader: DataLoader = None
    ):
        self.args = args
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.passage_with_idx = passage_with_idx
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.doc_dataloader = doc_dataloader
        self.test_dataloader = test_dataloader

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
        scaler = GradScaler()

        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        best_mrr1_scores = -1

        train_iterator = tqdm(range(int(self.args.num_train_epochs)), desc="Epoch")
        for epoch in train_iterator:    
            self.train_one_epoch(optimizer, scheduler, scaler, global_step)

            # doc_embdding = self.doc2embedding(self.p_encoder)
            mrr_scores = self.validation(self.p_encoder, self.q_encoder, self.doc_dataloader, self.valid_dataloader)

            if best_mrr1_scores <= mrr_scores:
                self.save_models(epoch)


    def save_models(self, epoch):
        base_path = "./encoder/"
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        self.p_encoder.save_pretrained(os.path.join(base_path, f"p_encoder_epoch{epoch}"))
        self.q_encoder.save_pretrained(os.path.join(base_path, f"q_encoder_epoch{epoch}"))


    @classmethod
    def doc2embedding(cls, p_encoder: nn.Module, doc_dataloader: DataLoader):
        doc_embedding = []
        with torch.no_grad():
            p_encoder.eval()
            for batch in tqdm(doc_dataloader):
                output = p_encoder(**batch)
                doc_embedding.append(output)
            doc_embedding = torch.cat(doc_embedding, dim=0)
        return doc_embedding


    @torch.no_grad()
    def validation(
        self, 
        p_encoder: nn.Module,
        q_encoder: nn.Module,
        doc_dataloader: DataLoader,
        query_dataloader: DataLoader
        ):
        p_encoder.eval()
        q_encoder.eval()
        all_embeddings = self.create_doc_embedding(p_encoder, doc_dataloader)
        # TODO
        top100_idx = []
        for batch in tqdm(query_dataloader):
            output = q_encoder(**batch)
            sim_scores = F.softmax(torch.matmul(output, all_embeddings.unsqueeze(1)), dim=1)
            sim_ranks = torch.argsort(sim_scores, dim=1)[:,:100].detach().cpu()
            top100_idx.append(sim_ranks)
        top100_idx = torch.cat(top100_idx, dim=1)
        return self.calculate_mrr(top100_idx, self.passage_with_idx)


    def train_one_epoch(
        self,
        optimizer,
        scheduler,
        scaler,
        global_step: int
    ):
        batch_size = self.args.per_device_train_batch_size
        with tqdm(self.train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:

                self.p_encoder.train()
                self.q_encoder.train()

                batch_size = len(batch[0]) # in case of drop_last = False
                targets = torch.arange(0, batch_size*(self.num_neg+1), (self.num_neg+1)).long() # ex) if num_neg==1, [0, 2, 4 ...] would be gt indices
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

                with autocast(enabled=True):
                    # (batch_size*(num_neg+1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size*, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    sim_scores = torch.matmul(q_outputs, p_outputs.T).squeeze() #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)

                tepoch.set_postfix(loss=f"{str(loss.item())}")

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                skip_lr_sched = (scale != scaler.get_scale())
                if not skip_lr_sched:
                    scheduler.step()

                # loss.backward()
                # optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                self.p_encoder.zero_grad()
                self.q_encoder.zero_grad()

                global_step += 1

                torch.cuda.empty_cache()

    @torch.no_grad()
    def create_doc_embedding(
        self,
        p_encoder: nn.Module,
        doc_dataloader: DataLoader
    ) -> torch.tensor:
        doc_embeddings = []
        p_encoder.eval()
        for batch in tqdm(doc_dataloader):
            bsz = batch.size(0)
            output = p_encoder(**batch).view(bsz, -1)
            doc_embeddings.append(output)
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        return doc_embeddings


    @staticmethod
    def inference(
        self,
        p_encoder: nn.Module = None,
        q_encoder: nn.Module = None,
        test_dataloader: DataLoader = None,
        docs_loader: DataLoader = None,
        hit_rate_k: int = 100
    ):
        if p_encoder is None:
            p_encoder = self.p_encoder
        if q_encoder is None:
            q_encoder = self.q_encoder
        if test_dataloader is None:
            test_dataloader = self.test_dataloader
        if doc_loader is None:
            doc_loader = self.doc_dataloader

        # get all doc embeddings
        all_embeddings = self.create_doc_embedding(p_encoder, doc_loader)

        # matmul to get similarity score
        all_rankings = []
        for batch in tqdm(test_dataloader):
            output = q_encoder(**batch)
            sim_scores = F.softmax(torch.matmul(output, all_embeddings.unsqueeze(0).T).squeeze(), dim=1).detach().cpu()
            ranking = torch.argsort(sim_scores)[:, :hit_rate_k]
            all_rankings.append(ranking)
        all_rankings.append(ranking)

        # calculate up to MRR@100
        pass
        

    def calculate_mrr(self, indices: torch.LongTensor, targets: torch.LongTensor) -> float:
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
        
        

class Seq2SeqTrainer(object):
    pass


def set_dataset(
    tokenizer,
    q_seqs,
    p_with_neg,
    num_neg: int = 1
) -> TensorDataset:
    assert len(p_with_neg) == len(q_seqs)*(num_neg+1), "wrong passage sequence lengths"
    q_seqs = tokenizer(
        q_seqs,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    p_seqs = tokenizer(
        p_with_neg,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    max_len = p_seqs["input_ids"].size(-1)
    p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg+1, max_len)
    p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg+1, max_len)
    p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg+1, max_len)

    return TensorDataset(
        p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
        q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
    )
    

def set_loader(dataset: TensorDataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size
    )

def dataloader_from_df(df: pd.DataFrame, columns: List[str]):
    pass


def doc2id():
    pass


def get_all_docs(path: str):
    pass




if __name__ == "__main__":

    # 1. create passage, query encoder & initial setting
    p_encoder = BertEncoder()
    q_encoder = BertEncoder()
    set_seed()

    #2. set following dataloaders
    #   train_dataloader           : q_seqs + p_with_neg
    #   valid_dataloader           : q_seqs + p_seqs(ground truth)
    #   passages with indices      : Dict[int, str] -> index + text
    #   (optional) doc_dataloader  : embedding for the whole passages encoded by p_encoder
    #   (optional) test_dataloader : contains only q_seqs
    # TODO

    # 3. create dual encoder trainer
    # TODO
    trainer = DualTrainer()

    # 4. train with validation
    # TODO

    # 5. inference
    # TODO


