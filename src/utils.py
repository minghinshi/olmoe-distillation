import os
from pathlib import Path

import torch as t
import torch.nn.functional as F
from datasets import IterableDataset, load_dataset
from dotenv import load_dotenv
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    BitsAndBytesConfig,
    OlmoeConfig,
    OlmoeForCausalLM,
)
from transformers.conversion_mapping import register_checkpoint_conversion_mapping
from transformers.models.gpt_neox.tokenization_gpt_neox import GPTNeoXTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP, OlmoeTopKRouter
from transformers.monkey_patching import clear_patch_mapping, register_patch_mapping

from dense_model import DenseOlmoeForCausalLM

MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"
DATASET_NAME = "allenai/tulu-v3.1-mix-preview-4096-OLMoE"

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

DENSE_MODEL_DIR = MODELS_DIR / "dense_model"
DISTILLED_MODEL_DIR = MODELS_DIR / "distilled_model"
ONNX_MODEL_PATH = MODELS_DIR / "onnx_model" / "model.onnx"

FINAL_CHECKPOINT_DIR = DISTILLED_MODEL_DIR / "checkpoint-2000"


class OlmoeQuantizableMoeBlock(nn.Module):
    """
    Implements OLMoE's MoE block using a `ModuleList`, so that it can be quantized.
    """

    def __init__(self, config: OlmoeConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.gate = OlmoeTopKRouter(config)

        # Define experts as a ModuleList of MLPs
        self.experts = nn.ModuleList(
            [OlmoeMLP(config) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: t.Tensor) -> t.Tensor:
        # Flatten (batch_size, sequence_length) dims
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Pass tokens through the router
        _, top_k_weights, top_k_index = self.gate(hidden_states)

        # Find activated experts
        with t.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = t.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        final_states = t.zeros_like(hidden_states)

        # For each activated expert:
        for expert_idx_tensor in expert_hit:
            # Convert to int for indexing
            expert_idx = int(expert_idx_tensor[0].item())
            assert expert_idx != self.num_experts

            # Get tokens that activate this expert
            top_k_pos, token_idx = t.where(expert_mask[expert_idx])
            hidden_state = hidden_states[token_idx]

            # Expert forward pass
            hidden_state = self.experts[expert_idx](hidden_state)

            # Weigh and add
            hidden_state = hidden_state * top_k_weights[token_idx, top_k_pos, None]
            final_states.index_add_(0, token_idx, hidden_state.to(final_states.dtype))

        # Reshape to original shape
        final_states = final_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_states


def load_olmoe() -> OlmoeForCausalLM:
    """
    Loads OLMoE and returns it unchanged.
    """
    return OlmoeForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cpu",
    )


def load_quantized_moe() -> OlmoeForCausalLM:
    """
    Loads OLMoE, replaces its expert modules with a version using `ModuleList`, and quantizes it in 4-bit.
    """
    # Monkey-patch the MoE block with a custom, quantizable version
    register_patch_mapping(
        mapping={
            "OlmoeSparseMoeBlock": OlmoeQuantizableMoeBlock,
        }
    )

    # OLMoE's checkpoint format stores expert weights separately as
    # idx.gate_proj.weight, idx.up_proj.weight, and idx.down_proj.weight
    # Originally, they will be merged into gate_up_proj and down_proj while loading
    # Now our custom MoE block fully follows checkpoint format, so we no longer need a mapping
    register_checkpoint_conversion_mapping(
        model_type_or_class_name="olmoe", mapping=[], overwrite=True
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=t.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Now we only use 7 GB instead of 14 GB!
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        quantization_config=quantization_config,
    )

    # Clean up
    clear_patch_mapping()
    assert isinstance(model, OlmoeForCausalLM)
    return model


def load_dense_model_incomplete() -> DenseOlmoeForCausalLM:
    """
    Loads a dense model with attention weights from OLMoE but MLP weights not set.

    Note: This will generate a warning from `transformers` that can be safely ignored.
    """
    return DenseOlmoeForCausalLM.from_pretrained(MODEL_NAME)


def load_dense_model() -> OlmoeForCausalLM:
    """
    Loads a dense model produced by extracting an expert from the MoE.
    """
    return DenseOlmoeForCausalLM.from_pretrained(
        DENSE_MODEL_DIR,
        device_map="cuda",
    )


def load_distilled_model() -> PeftModelForCausalLM:
    """
    Loads the distilled model as a PEFT model.
    """
    # Monkey-patch the MoE block with a dense MLP
    register_patch_mapping(
        mapping={
            "OlmoeSparseMoeBlock": OlmoeMLP,
        }
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        FINAL_CHECKPOINT_DIR,
        device_map="auto",
    )

    clear_patch_mapping()
    return model


def load_tokenizer() -> GPTNeoXTokenizer:
    """
    Loads OLMoE's tokenizer.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_sft_dataset() -> IterableDataset:
    """
    Loads the dataset used for supervised fine-tuning (SFT) of OLMoE.
    """
    return load_dataset(DATASET_NAME, split="train", streaming=True)


def load_and_check_env():
    """
    Loads environment variables and raise if you do not have all required variables.

    Required variables: `HF_TOKEN`
    """
    load_dotenv()
    assert os.getenv("HF_TOKEN"), "Please set HF_TOKEN in your .env file"


def test_drive(model, tokenizer):
    """
    Sends an LLM "Who are you?" and prints its response.
    """
    messages = [
        {
            "role": "user",
            "content": "Who are you?",
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    assert isinstance(inputs, BatchEncoding)
    inputs = inputs.to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)  # type: ignore
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))
