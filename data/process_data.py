import os
import re
import pandas as pd
import numpy as np
from datasets import load_from_disk
from typing import List, Union, Callable

from tqdm import tqdm
import argparse

from torch.utils.data import DataLoader, TensorDataset
from rank_bm25 import BM25Okapi

# TODO
# make all the data processing complete within a line of command
# make all dataset prepared with official ms-marco url


def get_bm25(corpus: List[str], tokenize_fn: Callable) -> BM25Okapi:
    tokenized_corpus = [tokenize_fn(txt) for txt in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def cleanse_corpus(corpus: Union[List, str]) -> Union[List, str]:
    if isinstance(corpus, List[str]):
        cleansed_corpus = [re.sub(f'[^- 0-9a-zA-Z]', ' ', text) for text in corpus]
        return cleansed_corpus
    else:
        return re.sub(f'[^- 0-9a-zA-Z]', ' ', corpus)


def cleanse_kor_corpus(corpus: Union[List[str], str]) -> Union[List[str], str]:
    if isinstance(corpus, List[str]):
        cleansed_corpus = [re.sub(f'[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]', ' ', text) for text in corpus]
        return cleansed_corpus
    else:
        return re.sub(f'[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]', ' ', corpus)


class InBatchNegative:
    def __init__(self, dataset_path: str, split: float = 0.2, num_neg: int = 1):
        self.split = split
        self.base_path = "./dataset"
        self.dataframe_path = os.path.join(self.base_path, dataset_path+".csv")
        if "tevatron" in dataset_path and not os.path.exists(self.dataframe_path):
            self.dataset = load_from_disk(os.path.join(dataset_path))["train"]
            self.dataframe = self.to_dataframe()
            self.dataframe.to_csv(self.dataframe_path, index=False)
        else:
            self.dataframe = pd.read_csv(self.dataframe_path)
        
    def to_dataframe(self):
        queries, pos, neg = [], [], []
        p_id, n_id = [], []
        for idx in tqdm(range(len(self.dataset))):
            queries.append(self.dataset[idx]["query"])
            pos.append(self.dataset[idx]["positive_passages"][0]["text"])
            neg.append(self.dataset[idx]["negative_passages"][0]["text"])
            p_id.append(int(self.dataset[idx]["positive_passages"][0]["docid"]))
            n_id.append(int(self.dataset[idx]["negative_passages"][0]["docid"]))
        return pd.DataFrame({
            "query": queries,
            "pos": pos,
            "neg": neg,
            "p_id": p_id,
            "n_id": n_id
        })
            
    def split_df(self):
        train_df = self.dataframe.sample(frac=0.8, random_state=0)
        valid_df = self.dataframe.drop(train_df.index)[["query", "pos", "p_id"]]
        return train_df, valid_df
        
    def p_with_neg(self, df: pd.DataFrame) -> List[str]:
        p_with_neg = []
        for _, (pos, neg) in tqdm(df[["pos", "neg"]].iterrows()):
            p_with_neg.extend([pos, neg])
        return p_with_neg
        
    def p_seqs(self, df) -> List[str]:
        p_seqs = []
        for _, (pos) in tqdm(df[["pos"]].iterrows()):
            p_seqs.append(pos)
        return p_seqs

    def q_seqs(self, df: pd.DataFrame) -> List[str]:
        q_seqs = []
        for _, (query) in tqdm(df[["query"]].iterrows()):
            q_seqs.append(query)
        return q_seqs

    @staticmethod
    def save_csv(self, df: pd.DataFrame):
        raise NotImplementedError



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="tevatron")
    args = parser.parse_args()
    dataset = InBatchNegative(args.data_name)
    # train_df, valid_df = dataset.split_df()
    # train_df.to_csv("./tevatron-train_df.csv", index=False)
    # valid_df.to_csv("./tevatron-valid_df.csv", index=False)

