# %% Setup
import torch as t
import torchinfo
from datasets import load_dataset
from torch import nn
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP

import utils

# %% Load your Hugging Face token
utils.load_and_check_env()

# %% Install the model
model = utils.load_olmoe()

# %% Test drive the model
tokenizer = utils.load_tokenizer()
utils.test_drive(model, tokenizer)

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
    model.model.layers[0].mlp,  # type: ignore
    input_size=(4, 42, 2048),
    dtypes=[t.bfloat16],
    col_names=("input_size", "output_size", "num_params"),
)

# %% Prepare some samples
dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

# %% Analyse expert activation frequency (unfinished)
model.eval()
device = model.device
dataset_iter = iter(dataset)

for _ in range(1):
    example = next(dataset_iter)

    # unsqueeze(0) to add the batch dimension
    input_ids = t.tensor(example["input_ids"]).unsqueeze(0).to(device)
    attention_mask = t.tensor(example["attention_mask"]).unsqueeze(0).to(device)

    with t.no_grad():
        output = model(input_ids, attention_mask, output_router_logits=True)

    # router_logits is a tuple of `num_layers` tensors
    # Each tensor has shape (`seq_len`, `num_experts`)
    # `num_layers` = 16, `num_experts` = 64
    router_logits: t.Tensor = output.router_logits
    print(router_logits[0].shape)
    print(router_logits)


# %% Create MLPs with weights from each layer's expert 0
# TODO: Pick the most frequently used expert instead
for i in range(len(model.model.layers)):
    layer = model.model.layers[i]
    new_mlp = OlmoeMLP(model.config)

    # Get weights from expert 0
    gate_up_proj: t.Tensor = layer.mlp.experts.gate_up_proj[0].detach().clone()  # type: ignore
    down_proj: t.Tensor = layer.mlp.experts.down_proj[0].detach().clone()  # type: ignore
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
model.save_pretrained(utils.DENSE_MODEL_PATH)
tokenizer.save_pretrained(utils.DENSE_MODEL_PATH)
