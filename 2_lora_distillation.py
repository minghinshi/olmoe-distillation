# LoRA vs. QLoRA comparison:
# https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-garden/lora-qlora

# %% Setup
from itertools import chain

import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import utils

# Number of tokens in each fine-tuning sample
SEQ_LEN = 2048

# Knowledge distillation temperature
T = 2.0

# %% Load your Hugging Face token
utils.load_and_check_env()

# %% Load the teacher
teacher = utils.load_quantized_moe()
print(teacher)

# %% Freeze the teacher and switch to eval mode
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

# %% Load the student
student = utils.load_dense_model()
print(student)

# %% Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(utils.MODEL_NAME)

# %% Set up low-rank adaptation (LoRA)
# https://huggingface.co/docs/peft/quicktour
# We're using one GPU, so DeepSpeed (for distributed training) can be ignored
# TODO: Investigate how to optimize the LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
)

student = get_peft_model(student, peft_config)

# Confirm trainable params is small
student.print_trainable_parameters()

# %% Load and preprocess the dataset
dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)


# For each batch of text, join all tokens into one big list
# and split it into chunks of 2048 tokens
def chunker(examples: dict[str, list]) -> dict[str, list]:
    chunked_ids: list[list[int]] = [[0] * SEQ_LEN]
    chunked_mask: list[list[int]] = [[0] * SEQ_LEN]
    token_idx = 0

    input_ids = chain.from_iterable(examples["input_ids"])
    attention_mask = chain.from_iterable(examples["attention_mask"])

    # Iterate over all tokens in all examples
    # TODO: Add end-of-text token between text
    for id, mask in zip(input_ids, attention_mask):
        chunked_ids[-1][token_idx] = id
        chunked_mask[-1][token_idx] = mask
        token_idx += 1

        # When chunk is full, add a new one
        if token_idx == SEQ_LEN:
            chunked_ids.append([0] * SEQ_LEN)
            chunked_mask.append([0] * SEQ_LEN)
            token_idx = 0

    # Drop the non-full chunk
    chunked_ids.pop()
    chunked_mask.pop()
    return {"input_ids": chunked_ids, "attention_mask": chunked_mask}


dataset = dataset.map(
    chunker,
    batched=True,
    remove_columns=["text", "timestamp", "url"],
)

# %% Create a data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# %% Create a trainer
class DistillationTrainer(Trainer):
    def __init__(self, teacher: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Forward passes
        student_output = model(**inputs)
        student_logits = student_output.logits
        teacher_logits = self.teacher(**inputs).logits

        # Compute loss (KL divergence)
        # Use temperature > 1 for better performance
        student_logprobs = F.log_softmax(student_logits / T, dim=-1)
        teacher_logprobs = F.log_softmax(teacher_logits / T, dim=-1)
        vocab_size = student_logprobs.shape[-1]

        loss = F.kl_div(
            student_logprobs.view(-1, vocab_size),
            teacher_logprobs.view(-1, vocab_size),
            reduction="batchmean",
            log_target=True,
        ) * (T * T)

        return (loss, student_output) if return_outputs else loss


# https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
training_args = TrainingArguments(
    output_dir="./models/distilled_model",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    max_steps=500,
    learning_rate=2e-4,
    gradient_accumulation_steps=8,
    # Requires Ampere or newer GPUs
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=4,
)

trainer = DistillationTrainer(
    teacher=teacher,
    model=student,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# %% Time to train!
trainer.train()

# Observations:

# The initial training loss is around 100.
# If we print the loss in `compute_loss`, we get around 10-15.
# `vocab_size` is about 50000 and ln(50000) is about 10.8,
# so the dense model is at first performing no better than random.

# Extracting expert 0 or averaging expert weights gave similar initial loss.
