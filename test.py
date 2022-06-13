import numpy as np
import time
from tqdm import tqdm
from pprint import pprint
from typing import NoReturn, List, Union

import torch
import torch.nn as nn
from torch import Tensor as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from utils.calc_utils import calculate_mrr

MaxMRRRank = 100

# TODO: create class for end-to-end inference
# TODO: implement raw inference and faiss inference


@torch.no_grad()
def inference(
    p_encoder: nn.Module, 
    q_encoder: nn.Module, 
    all_docs: Union[List[str], DataLoader], 
    q_seqs: List[str],
    q_gt_indices: List[int] = None
    ):
    p_encoder.eval()
    q_encoder.eval()
    if isinstance(all_docs, List[str]):
        docs_dataloader: DataLoader = get_docs_dataloader(all_docs)
    else:
        docs_dataloader = all_docs
    docs_embedding: T = get_docs_embeddings(p_encoder, docs_dataloader)
    q_seqs_loader: DataLoader = get_query_loader(q_seqs)









# def raw_inference(retriever: DenseRetriever, valid_dataset: Dataset)->NoReturn:
#     idx = np.random.randint(len(valid_dataset))
#     query = valid_dataset[idx]["question"]

#     print(f"[Search Query] \n{query}\n")
#     print(f"[Ground Truth Passage]")
#     pprint(valid_dataset[idx]['context'])

#     start = time.time()
#     results = retriever.get_relevant_doc(query=query, k=3)
#     end = time.time()

#     indices = results.tolist()
#     print(f"Took {end-start}s Retrieving For Single Question")
#     for i, idx in enumerate(indices):
#         print(f"Top-{i + 1}th Passage (Index {idx})")
#         pprint(retriever.valid_corpus[idx])


# def faiss_inference(
#     retriever: DenseRetriever,
#     valid_dataset: Dataset
# )->NoReturn:
#     ds_with_embeddings = valid_dataset.map(lambda example: {"embeddings": retriever.p_encoder(**retriever.tokenizer(example["context"],
#                                                                                                 padding="max_length",
#                                                                                                 truncation=True,
#                                                                                                 return_tensors="pt").to(device).squeeze().cpu().numpy())})
#     ds_with_embeddings.add_faiss_index(column='embeddings', device=0)

#     # Validation Set의 전체 질문들에 대한 embedding 구해놓
#     question_embedding = []
#     for question in tqdm(valid_dataset["question"]):
#         tokenized = retriever.tokenizer(question, padding="max_length", truncation=True, return_tensors="pt").to(retriever.device)
#         encoded = retriever.q_encoder(**tokenized).squeeze().detach().cpu().numpy()
#         question_embedding.append(encoded)

#     question_embedding = np.array(question_embedding)

#     start = time.time()
#     # 여러 개 한번에 retrieval
#     scores, retrieved_examples = ds_with_embeddings.get_nearest_examples_batch('embeddings', question_embedding, k=3)
#     end = time.time()
#     print(f"Took {end-start}s Retreiving the Whole Validation Dataset")
#     pprint(retrieved_examples[0]["context"])
#     print(retrieved_examples[0]["question"])

#     # 개별 query retrieval
#     retrieved = ds_with_embeddings.get_nearest_examples('embeddings', question_embedding[0], k=3)
#     pprint(retrieved[1]["context"])

#     idx = np.random.randint(len(valid_dataset))
#     print(f"[Question]\n{valid_dataset['question'][idx]}")
#     print(f"[Ground Truth]\n{valid_dataset['context'][idx]}")
#     retrieved_docs = retrieved_examples[idx]["context"]
#     for i, doc in enumerate(retrieved_docs):
#         print(f"Top-{i + 1}th Passage")
#         pprint(doc)

if __name__ == "__main__":
    valid_dataset = load_dataset("klue", "mrc", split="validation")

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
