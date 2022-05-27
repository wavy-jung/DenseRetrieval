import numpy as np
import time
from tqdm import tqdm
from pprint import pprint
from typing import NoReturn

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from model.dpr import DenseRetriever

MaxMRRRank = 100


def compute_metrics(qids_to_relevant_documentids, qids_to_ranked_candidate_documents, exclude_qids):
    """Compute MRR metric
    Args:    
    p_qids_to_relevant_documentids (dict): dictionary of query-document mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_documents (dict): dictionary of query-document candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    qids_with_relevant_documents = 0
    ranking = []
    
    for qid in qids_to_ranked_candidate_documents:
        if qid in qids_to_relevant_documentids and qid not in exclude_qids:
            ranking.append(0)
            target_pid = qids_to_relevant_documentids[qid]
            candidate_pid = qids_to_ranked_candidate_documents[qid]
            for i in range(0, MaxMRRRank):
                if candidate_pid[i][0] in target_pid:
                    MRR += 1/(i + 1)
                    ranking.pop()
                    ranking.append(i+1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")
    
    MRR = MRR/len(qids_to_relevant_documentids)
    all_scores['MRR @100'] = MRR
    all_scores['QueriesRanked'] = len(set(qids_to_ranked_candidate_documents)-exclude_qids)
    return all_scores


def raw_inference(retriever: DenseRetriever, valid_dataset: Dataset)->NoReturn:
    idx = np.random.randint(len(valid_dataset))
    query = valid_dataset[idx]["question"]

    print(f"[Search Query] \n{query}\n")
    print(f"[Ground Truth Passage]")
    pprint(valid_dataset[idx]['context'])

    start = time.time()
    results = retriever.get_relevant_doc(query=query, k=3)
    end = time.time()

    indices = results.tolist()
    print(f"Took {end-start}s Retrieving For Single Question")
    for i, idx in enumerate(indices):
        print(f"Top-{i + 1}th Passage (Index {idx})")
        pprint(retriever.valid_corpus[idx])


def faiss_inference(
    retriever: DenseRetriever,
    valid_dataset: Dataset
)->NoReturn:
    ds_with_embeddings = valid_dataset.map(lambda example: {"embeddings": retriever.p_encoder(**retriever.tokenizer(example["context"],
                                                                                                padding="max_length",
                                                                                                truncation=True,
                                                                                                return_tensors="pt").to(device).squeeze().cpu().numpy())})
    ds_with_embeddings.add_faiss_index(column='embeddings', device=0)

    # Validation Set의 전체 질문들에 대한 embedding 구해놓
    question_embedding = []
    for question in tqdm(valid_dataset["question"]):
        tokenized = retriever.tokenizer(question, padding="max_length", truncation=True, return_tensors="pt").to(retriever.device)
        encoded = retriever.q_encoder(**tokenized).squeeze().detach().cpu().numpy()
        question_embedding.append(encoded)

    question_embedding = np.array(question_embedding)

    start = time.time()
    # 여러 개 한번에 retrieval
    scores, retrieved_examples = ds_with_embeddings.get_nearest_examples_batch('embeddings', question_embedding, k=3)
    end = time.time()
    print(f"Took {end-start}s Retreiving the Whole Validation Dataset")
    pprint(retrieved_examples[0]["context"])
    print(retrieved_examples[0]["question"])

    # 개별 query retrieval
    retrieved = ds_with_embeddings.get_nearest_examples('embeddings', question_embedding[0], k=3)
    pprint(retrieved[1]["context"])

    idx = np.random.randint(len(valid_dataset))
    print(f"[Question]\n{valid_dataset['question'][idx]}")
    print(f"[Ground Truth]\n{valid_dataset['context'][idx]}")
    retrieved_docs = retrieved_examples[idx]["context"]
    for i, doc in enumerate(retrieved_docs):
        print(f"Top-{i + 1}th Passage")
        pprint(doc)

if __name__ == "__main__":
    valid_dataset = load_dataset("klue", "mrc", split="validation")

    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
