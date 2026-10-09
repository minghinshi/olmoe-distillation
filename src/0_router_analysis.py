# %% Setup
import torch as t
import torch.nn.functional as F
from tqdm import tqdm

import utils

# %% Load the model
model = utils.load_quantized_moe()

# %% Load the tokenizer
tokenizer = utils.load_tokenizer()


# %% Prepare some samples
def tokenize(examples: dict[str, list]):
    return tokenizer.apply_chat_template(examples["messages"])


dataset = utils.load_sft_dataset()
dataset = dataset.map(tokenize)

# %% Analyse expert activation frequency
# For more granular information, we calculate the sum of weights
# that the router assigned to each expert
device = model.device
num_layers = model.config.num_hidden_layers
num_experts = model.config.num_experts

dataset_iter = iter(dataset)
total_weights = t.zeros((num_layers, num_experts)).to(device)

for _ in tqdm(range(100)):
    example = next(dataset_iter)

    # unsqueeze(0) to add the batch dimension
    input_ids = t.tensor(example["input_ids"]).unsqueeze(0).to(device)
    attention_mask = t.tensor(example["attention_mask"]).unsqueeze(0).to(device)

    with t.no_grad():
        output = model(input_ids, attention_mask, output_router_logits=True)

    # router_logits has shape (`num_layers`, `seq_len`, `num_experts`)
    router_logits = t.stack(output.router_logits)
    router_weights = F.softmax(router_logits, dim=-1)
    total_weights += t.sum(router_weights, dim=1)

# %% For each layer, find the expert with the largest total weight
importance = total_weights / t.sum(total_weights, dim=-1, keepdim=True) * num_experts
print(t.max(importance, dim=-1))

# `importance` tells how important each expert is in that layer
# 1 = baseline, 2 = assigned 2x the weight by the router on average, etc.
# It doesn't change the result of the analysis, but is more interpretable
