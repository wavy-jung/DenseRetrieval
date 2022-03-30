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


def train(
    p_encoder: nn.Module,
    q_encoder: nn.Module,
    train_dataloader: DataLoader,
    args: TrainingArguments,
    num_neg: int = 1,
)->NoReturn:

    set_train(args)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params": [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )

    # Start training!
    global_step = 0

    p_encoder.zero_grad()
    q_encoder.zero_grad()
    torch.cuda.empty_cache()

    train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
    for _ in train_iterator:    
        batch_size = args.per_device_train_batch_size
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for batch in tepoch:

                p_encoder.train()
                q_encoder.train()

                batch_size = len(batch[0]) # 마지막 batch에서 sample수가 다른 경우 dim. error 발생 방지하기 위해 batch_size 업데이트
                targets = torch.arange(0, batch_size*(num_neg+1), (num_neg+1)).long() # ex) num_neg=1인 경우, 정답은 [0, 2, 4 ...] 인덱스
                targets = targets.to(args.device)
            
                p_inputs = {
                    "input_ids": batch[0].view(batch_size * (num_neg + 1), -1).to(args.device),
                    "attention_mask": batch[1].view(batch_size * (num_neg+1), -1).to(args.device),
                    "token_type_ids": batch[2].view(batch_size * (num_neg+1), -1).to(args.device)
                }
            
                q_inputs = {
                    "input_ids": batch[3].to(args.device),
                    "attention_mask": batch[4].to(args.device),
                    "token_type_ids": batch[5].to(args.device)
                }

                # (batch_size*(num_neg+1), emb_dim)
                p_outputs = p_encoder(**p_inputs)["pooler_output"]
                # (batch_size*, emb_dim)
                q_outputs = q_encoder(**q_inputs)["pooler_output"]

                # Calculate similarity score & loss
                sim_scores = torch.matmul(q_outputs, p_outputs.T).squeeze() #(batch_size, num_neg + 1)
                sim_scores = sim_scores.view(batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)
                tepoch.set_postfix(loss=f"{str(loss.item())}")

                loss.backward()
                optimizer.step()
                scheduler.step()

                p_encoder.zero_grad()
                q_encoder.zero_grad()

                global_step += 1

                torch.cuda.empty_cache()


def set_train(args: TrainingArguments)->NoReturn:
    set_seed()
    args = TrainingArguments()

    train_dataset = load_dataset("klue", "mrc", split="train")
    valid_dataset = load_dataset("klue", "mrc", split="validation")

    model_checkpoint = "klue/bert-base"

    bm25_df = pd.read_csv("klue_bm25.csv")
    bm25_df["negatives"] = bm25_df["negatives"].apply(lambda lst: eval(lst))

    retriever = DenseRetriever(model_checkpoint)

    train(
        p_encoder=retriever.p_encoder,
        q_encoder=retriever.q_encoder,
        train_dataloader=None,
        num_neg=1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arguments")
    parser.add_argument("--")

    retriever = DenseRetriever("klue/bert-base")
    train(
        p_encoder=retriever.p_encoder,
        q_encoder=retriever.q_encoder,
        #TODO
    )