# LoRA vs. QLoRA comparison:
# https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-garden/lora-qlora

# %% Setup
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import utils

# Target context window size of the student
# Larger values give the student a longer context window
# but requires more VRAM for distillation
CONTEXT_LEN = 2048

# Knowledge distillation temperature
T = 2.0

# Whether we're doing the real multi-hour run or a test run
REAL_RUN = False
TRAINING_STEPS = 2000 if REAL_RUN else 10
LOGGING_STEPS = 20 if REAL_RUN else 1

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
tokenizer = utils.load_tokenizer()

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
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
)

student = get_peft_model(student, peft_config)

# Confirm trainable params is small
student.print_trainable_parameters()

# %% Load and preprocess the dataset
# We'll use the same dataset used for SFT of OLMoE
DATASET_NAME = "allenai/tulu-v3.1-mix-preview-4096-OLMoE"
dataset = load_dataset(DATASET_NAME, split="train", streaming=True)


def tokenize(examples: dict[str, list]):
    return tokenizer.apply_chat_template(
        examples["messages"],
        add_generation_prompt=False,
        padding=True,
        truncation=True,
        max_length=CONTEXT_LEN,
    )


dataset = dataset.map(tokenize, batched=True)
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
    output_dir="../models/distilled_model",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    max_steps=TRAINING_STEPS,
    learning_rate=2e-4,
    gradient_accumulation_steps=8,
    # Requires Ampere or newer GPUs
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=LOGGING_STEPS,
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
