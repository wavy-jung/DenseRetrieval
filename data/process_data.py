import os
import re
import pandas as pd
import numpy as np
from datasets import load_from_disk
from typing import List, Union, Optional, Any, Callable, Tuple

from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from rank_bm25 import BM25Okapi


def get_bm25(corpus: List[str], tokenize_fn: Callable)->BM25Okapi:
    tokenized_corpus = [tokenize_fn(txt) for txt in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def cleanse_corpus(corpus: Any[List, str]) -> Any[List, str]:
    if isinstance(corpus, List[str]):
        cleansed_corpus = [re.sub(f'[^- 0-9a-zA-Z]', ' ', text) for text in corpus]
        return cleansed_corpus
    else:
        return re.sub(f'[^- 0-9a-zA-Z]', ' ', corpus)


def cleanse_kor_corpus(corpus: Any[List, str]) -> Any[List, str]:
    if isinstance(corpus, List[str]):
        cleansed_corpus = [re.sub(f'[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]', ' ', text) for text in corpus]
        return cleansed_corpus
    else:
        return re.sub(f'[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]', ' ', corpus)


class InBatchNegative:
    def __init__(self, dataset_path: str, split: float = 0.2, num_neg: int = 1):
        self.split = split
        self.base_path = "./"
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


# TODO
def prepare_in_batch_negative(
    dataset=None,
    num_neg=1,
    tokenizer=None,
    args=None
)->DataLoader:

    # 1. In-Batch-Negative 만들기
    # CORPUS를 np.array로 변환해줍니다.
    corpus = np.array(list(set([example for example in dataset["context"]])))

    p_with_neg = []
    # for idx, c in enumerate(tqdm(dataset["context"], total=len(dataset["context"]), desc="in-batch negative")):
    #     p_with_neg.append(c)
    #     if num_neg == 0:
    #         continue
    #     neg_documents = src_negatives["negatives"][idx]
    #     p_with_neg.extend(neg_documents[:num_neg])      

    # 2. (Question, Passage) 데이터셋 만들어주기
    q_seqs = tokenizer(
        dataset["question"],
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

    train_dataset = TensorDataset(
        p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
        q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size
    )

    return train_dataloader