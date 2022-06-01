from dataclasses import field, dataclass
import torch

avail_device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingArguments:
    num_train_epochs: int = field(
        default=3, 
        metadata={"help": "num train epochs"}
        )
    learning_rate: float = field(
        default=5e-5, 
        metadata={"help": "initial learning rate"}
        )
    per_device_train_batch_size: int = field(
        default=4, 
        metadata={"help": "train batch size (num queries per batch)"}
        )
    per_device_eval_batch_size: int = field(
        default=4, 
        metadata={"help": "eval batch size (num queries per batch)"}
        )
    device: str = field(
        default=avail_device, 
        metadata={"help": "train in which device"}
        )
    warmup_steps: int = field(
        default=0, 
        metadata={"help": "steps for learning rate warmup"}
        )
    weight_decay: float = field(
        default=0.01, 
        metadata={"help": "weight decay"}
        )
    adam_epsilon: float = field(
        default=1e-8, 
        metadata={"help": "adam epsilon"}
        )
    gradient_accumulation_steps: int = field(
        default=1, 
        metadata={"help": "gradient accumulation step"}
        )
