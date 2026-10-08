import torch as t
from datasets import load_dataset

import utils

# %% Load the model
model = utils.load_olmoe()

# %% Load the tokenizer
tokenizer = utils.load_tokenizer()

# %% Prepare some samples
dataset = load_dataset("allenai/c4", "en", streaming=True, split="train")
dataset = dataset.map(lambda x: tokenizer(x["text"]), batched=True)

# %% Analyse expert activation frequency
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
