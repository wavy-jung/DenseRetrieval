import os
import numpy as np
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)
from rank_bm25 import BM25Okapi



class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
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
        
        pooled_output = outputs[1]
        return pooled_output


class DenseRetrieval:
    def __init__(self,
        args,
        dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder,
        src_negatives=None,
        valid_dataset=None
    ):

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg
        assert num_neg in [0, 1, 2], "num_neg should be selected among [0, 1, 2]"

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.bm25 = None
        self.src_negatives = src_negatives
        self.valid_dataset = valid_dataset
        self.valid_corpus = None

        self.prepare_in_batch_negative(num_neg=num_neg)
        self._set_dataloader(valid_dataset=valid_dataset)

    def prepare_in_batch_negative(self,
        dataset=None,
        valid_dataset=None,
        num_neg=1,
        tokenizer=None
    ):

        if dataset is None:
            dataset = self.dataset

        if valid_dataset is None:
            valid_dataset = self.valid_dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset["context"]])))
 
        if self.src_negatives is None: 
            raise NotImplementedError("bm25 real-time batch not implemented")
        else:
            assert list(dataset["question"])==list(self.src_negatives["question"]), "wrong negative source"
        p_with_neg = []
        for idx, c in enumerate(tqdm(dataset["context"], total=len(dataset["context"]), desc="in-batch negative")):
            p_with_neg.append(c)
            if num_neg == 0:
                continue
            neg_documents = self.src_negatives["negatives"][idx]
            p_with_neg.extend(neg_documents[:num_neg])      

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

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size
        )


    def _set_dataloader(self, tokenizer=None, valid_dataset=None):
        if tokenizer is None:
            tokenizer = self.tokenizer

        if valid_dataset is None:
            valid_dataset = self.valid_dataset
        self.valid_corpus = list(set(valid_dataset["context"]))

        valid_seqs = tokenizer(
            self.valid_corpus,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
            valid_seqs["token_type_ids"]
        )
        self.passage_dataloader = DataLoader(
            passage_dataset,
            batch_size=self.args.per_device_eval_batch_size
        )        


    def _set_bm25(self, 
            pretrained_tokenizer="monologg/koelectra-base-v3-discriminator",
            dataset=None
        ):
        if isinstance(self.bm25, BM25Okapi):
            return
        else:
            if dataset is None:
                dataset = self.dataset["context"]
            self.bm25_tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
            tokenized_corpus = [self.bm25_tokenizer.tokenize(text) for text in tqdm(dataset, desc="tokenizing corpus")]
            self.bm25 = BM25Okapi(tqdm(tokenized_corpus, desc="bm25 setting"))

    
    def _get_bm25_scores(self, query):
        scores = self.bm25.get_scores(query)
        return scores


    def _check_valid_negative(self, gt_passage, negative_passage, answers):
        # ground_truth_passage와 negative_sample_passage 일치하는지 확인
        if gt_passage == negative_passage:
            return False
        # 정답들 중 가장 길이가 긴 text만 확인
        is_valid = self._check_valid_neg_ans(negative_passage, answers)
        return is_valid


    def _check_valid_neg_ans(self, negative_passage, answers):
        answer = sorted(answers, key=lambda text: -len(text))[0]
        if answer in negative_passage:
            return False
        return True


    def train(self, args=None):
        if args is None:
            args = self.args

        # Optimizer & Scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        for _ in train_iterator:
            batch_size = args.per_device_train_batch_size
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    batch_size = len(batch[0]) # 마지막 batch에서 sample수가 다른 경우 dim. error 발생 방지하기 위해 batch_size 업데이트
                    targets = torch.arange(0, batch_size*(self.num_neg+1), (self.num_neg+1)).long() # ex) num_neg=1인 경우, 정답은 [0, 2, 4 ...] 인덱스
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

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

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs


    def get_relevant_doc(self,
        query,
        k=1,
        args=None,
        p_encoder=None,
        q_encoder=None,
        valid_dataset=None
    ):
    
        if args is None:
            args = self.args
            batch_size = args.per_device_eval_batch_size

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        if valid_dataset is None:
            valid_dataset = self.valid_dataset
        else:
            self._set_dataloader(valid_dataset=valid_dataset)

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(args.device)
            q_emb = q_encoder(**q_seqs_val).to("cpu")  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
                p_emb = p_encoder(**p_inputs).to("cpu")
                p_embs.append(p_emb)
        
        # (num_passage, emb_dim)
        p_embs = torch.cat(
            [torch.stack(p_embs[:-1]).view(batch_size*(len(self.passage_dataloader.dataset)//batch_size), -1), p_embs[-1]]
        ).view(len(self.passage_dataloader.dataset), -1) # 마지막 전체 loader가 batch_size로 나누어 떨어지지 않을 때 단순히 stack을 하면 dim. error 발생

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1)) # (num_query, num_passage)
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()

        return rank[:k]


    def get_relevant_doc_bulk():
        pass