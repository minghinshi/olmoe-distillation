# %% Setup
import os

import torch as t
import torchinfo
from dotenv import load_dotenv
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

import utils

# %% Load your Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set HF_TOKEN in your .env file"

# %% Install the model
model = AutoModelForCausalLM.from_pretrained(utils.MODEL_NAME, device_map="auto")

# %% Test drive the model
tokenizer = AutoTokenizer.from_pretrained(utils.MODEL_NAME)
messages = [{"role": "user", "content": "Who are you?"}]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)  # type: ignore
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))

# %% Print the model architecture
print(model)

# %% Print additional useful information using torchinfo
torchinfo.summary(
    model,
    input_size=(4, 42),
    dtypes=[t.long],
    col_names=("input_size", "output_size", "num_params"),
)

# %% Analyze the MoE block to be replaced using torchinfo
torchinfo.summary(
    model.model.layers[0].mlp,
    input_size=(4, 42, 2048),
    dtypes=[t.bfloat16],
    col_names=("input_size", "output_size", "num_params"),
)

# %% Create MLPs with weights from each layer's expert 0
# TODO: Pick the most frequently used expert instead
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    new_mlp = OlmoeMLP(model.config)

    # Get weights from expert 0
    gate_up_proj: t.Tensor = layer.mlp.experts.gate_up_proj[0].detach().clone()
    down_proj: t.Tensor = layer.mlp.experts.down_proj[0].detach().clone()
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
    dtypes=[t.long],
    col_names=("input_size", "output_size", "num_params"),
)

# %% Save the model for the next file
model.save_pretrained("./models/dense_model")
tokenizer.save_pretrained("./models/dense_model")
