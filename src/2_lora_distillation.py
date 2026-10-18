# %% Setup
import torch as t
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

import utils

utils.load_and_check_env()

# Knowledge distillation temperature
T = 2.0

# Whether we're doing the real multi-hour run or a test run
DEBUG = True
TRAINING_STEPS = 10 if DEBUG else 2000
LOGGING_STEPS = 1 if DEBUG else 20

# %% Load the teacher
teacher = utils.load_quantized_moe()

# %% Freeze the teacher and switch to eval mode
teacher.eval()
for param in teacher.parameters():
    param.requires_grad = False

# %% Load the student
student = utils.load_dense_model()

# %% Attach LoRA adapters to student
# Training configuration from the following guide:
# https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
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

peft_student = get_peft_model(student, peft_config)
peft_student.print_trainable_parameters()

# %% Load the tokenizer
tokenizer = utils.load_tokenizer()

# %% Load the dataset
# We'll use the same dataset used for SFT of OLMoE
dataset = utils.load_sft_dataset()


# %% Create a trainer
class SFTDistillationTrainer(SFTTrainer):
    def __init__(self, teacher: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict,
        return_outputs=False,
        num_items_in_batch=None,
    ) -> t.Tensor:

        # Mask of trainable tokens
        mask: t.Tensor = inputs["labels"][:, 1:] != -100

        # Remove labels in model input
        # so that they don't compute loss by themselves
        real_input = inputs.copy()
        del real_input["labels"]

        # Number of tokens in this gradient accumulation window
        # Clamped to 1 to avoid division by zero
        n_tokens = mask.sum() if num_items_in_batch is None else num_items_in_batch
        n_tokens = t.as_tensor(n_tokens).clamp(min=1)

        # Student forward pass
        student_output = model(**real_input)
        student_logits: t.Tensor = student_output.logits[:, :-1][mask]

        # Teacher forward pass
        with t.no_grad():
            teacher_logits: t.Tensor = self.teacher(**real_input).logits[:, :-1][mask]

        # KL-divergence loss
        # Use temperature > 1 for better performance
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.log_softmax(teacher_logits / T, dim=-1),
            reduction="sum",
            log_target=True,
        )
        kl_loss = kl_loss * T * T / n_tokens

        # Cross-entropy loss
        labels: t.Tensor = inputs["labels"][:, 1:][mask]
        logprobs = F.log_softmax(student_logits, dim=-1)
        label_logprobs = logprobs.gather(-1, labels.unsqueeze(-1))
        ce_loss = -label_logprobs.sum() / n_tokens

        if DEBUG:
            print(f"KL: {kl_loss:.3f}, CE: {ce_loss:.3f}")

        return kl_loss + ce_loss


training_args = SFTConfig(
    output_dir=str(utils.DISTILLED_MODEL_DIR),
    per_device_train_batch_size=2,
    num_train_epochs=1,
    max_steps=TRAINING_STEPS,
    learning_rate=2e-4,
    gradient_accumulation_steps=8,
    # Requires Ampere or newer GPUs
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=LOGGING_STEPS,
    assistant_only_loss=True,
)

trainer = SFTDistillationTrainer(
    teacher=teacher,
    model=peft_student,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# %% Time to train!
trainer.train()
