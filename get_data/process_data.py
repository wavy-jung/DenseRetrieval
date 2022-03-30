import re
from typing import List, Union, Optional, Any, Callable
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, TensorDataset
from rank_bm25 import BM25Okapi


def get_bm25(corpus: List[str], tokenize_fn: Callable)->BM25Okapi:
    tokenized_corpus = [tokenize_fn(txt) for txt in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def cleanse_corpus(corpus: Any[List, str])->Any[List, str]:
    if isinstance(corpus, List[str]):
        cleansed_corpus = [re.sub(f'[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]', ' ', text) for text in corpus]
        return cleansed_corpus
    else:
        return re.sub(f'[^- ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z]', ' ', corpus)


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