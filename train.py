import argparse
import pandas as pd

from src.utils import set_seed

from model.dpr import BertEncoder, DenseRetrieval
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset


def train():
    set_seed()
    args = TrainingArguments(
                            output_dir="dense_retireval",
                            evaluation_strategy="epoch",
                            learning_rate=5e-5,
                            per_device_train_batch_size=4,
                            per_device_eval_batch_size=4,
                            num_train_epochs=3,
                            weight_decay=0.01)

    train_dataset = load_dataset("klue", "mrc", split="train")
    valid_dataset = load_dataset("klue", "mrc", split="validation")

    model_checkpoint = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)

    bm25_df = pd.read_csv("klue_bm25.csv")
    bm25_df["negatives"] = bm25_df["negatives"].apply(lambda lst: eval(lst))

    retriever = DenseRetrieval(
        args=args,
        dataset=train_dataset,
        num_neg=1,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        src_negatives=bm25_df,
        valid_dataset=valid_dataset
    )

    retriever.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arguments")
    parser.add_argument("--")
    train()