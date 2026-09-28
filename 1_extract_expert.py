# %% Setup
import os

import torch
import torchinfo
from dotenv import load_dotenv
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

# %% Load your Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set HF_TOKEN in your .env file"

# %% Install the model
MODEL_NAME = "allenai/OLMoE-1B-7B-0924-Instruct"

# Set device_map to "cuda" to load in GPU
# If "auto", the program thinks the model doesn't fit in GPU memory
# so it tries to load it in CPU, causing a downstream error
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda")

# %% Test drive the model
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924-Instruct")
messages = [{"role": "user", "content": "Who are you?"}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

# %% Print the model architecture
print(model)

# %% Print additional useful information using torchinfo
torchinfo.summary(
    model,
    input_size=(4, 42),
    dtypes=[torch.long],
    col_names=("input_size", "output_size", "num_params"),
)

# %% Analyze the MoE block to be replaced using torchinfo
torchinfo.summary(
    model.model.layers[0].mlp,
    input_size=(4, 42, 2048),
    dtypes=[torch.bfloat16],
    col_names=("input_size", "output_size", "num_params"),
)

# Here's the original MoE block for reference
# class OlmoeSparseMoeBlock(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.gate = OlmoeTopKRouter(config)
#         self.experts = OlmoeExperts(config)

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         batch_size, sequence_length, hidden_dim = hidden_states.shape
#         hidden_states = hidden_states.view(-1, hidden_dim)
#         _, top_k_weights, top_k_index = self.gate(hidden_states)
#         final_hidden_states = self.experts(hidden_states, top_k_index, top_k_weights).reshape(
#             batch_size, sequence_length, hidden_dim
#         )
#         return final_hidden_states

# @use_experts_implementation
# class OlmoeExperts(nn.Module):
#     """Collection of expert weights stored as 3D tensors."""

#     def __init__(self, config: OlmoeConfig):
#         super().__init__()
#         self.num_experts = config.num_local_experts
#         self.hidden_dim = config.hidden_size
#         self.intermediate_dim = config.intermediate_size
#         self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
#         self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         top_k_index: torch.Tensor,
#         top_k_weights: torch.Tensor,
#     ) -> torch.Tensor:
#         final_hidden_states = torch.zeros_like(hidden_states)
#         with torch.no_grad():
#             expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
#             expert_mask = expert_mask.permute(2, 1, 0)
#             expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

#         for expert_idx in expert_hit:
#             expert_idx = expert_idx[0]
#             if expert_idx == self.num_experts:
#                 continue
#             top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
#             current_state = hidden_states[token_idx]
#             gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
#             current_hidden_states = self.act_fn(gate) * up
#             current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
#             current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
#             final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

#         return final_hidden_states

# Note these two lines:
# gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
# current_hidden_states = self.act_fn(gate) * up

# This implements an activation function called "SwiGLU".
# SwiGLU(x, W, V) = Swish(xW + b) ⊗ (xV + c)
# The matrices W and V are packed into self.gate_up_proj above.

# hidden_dim = 2048 while intermediate_dim = 1024,
# so up and down projection are misnomers in this case.

# %% Create MLPs with weights from each layer's expert 0
# TODO: Pick the most frequently used expert instead
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    new_mlp = OlmoeMLP(model.config)

    # Get weights from expert 0
    gate_up_proj: torch.Tensor = layer.mlp.experts.gate_up_proj[0].detach().clone()
    down_proj: torch.Tensor = layer.mlp.experts.down_proj[0].detach().clone()
    gate_proj, up_proj = gate_up_proj.chunk(2)

    new_mlp.gate_proj.weight = nn.Parameter(gate_proj)
    new_mlp.up_proj.weight = nn.Parameter(up_proj)
    new_mlp.down_proj.weight = nn.Parameter(down_proj)

    # Replace the MLP module
    model.set_submodule(f"model.layers.{i}.mlp", new_mlp)

# %% Print the new model architecture
print(model)

# %% Print torchinfo
# Params drop from 6.9B to 575M
torchinfo.summary(
    model,
    input_size=(4, 42),
    dtypes=[torch.long],
    col_names=("input_size", "output_size", "num_params"),
)

# %% Save the model for the next file
model.save_pretrained("./models/dense_model")
tokenizer.save_pretrained("./models/dense_model")
