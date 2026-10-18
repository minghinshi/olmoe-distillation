# %% Setup
import torch as t
import torch.nn.functional as F
from tqdm import tqdm

import utils

utils.load_and_check_env()

# Number of examples to use for estimating expert frequency
N_SAMPLES = 100

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
k = model.config.num_experts_per_tok

dataset_iter = iter(dataset)
total_weights = t.zeros((num_layers, num_experts)).to(device)
total_tokens = 0

for _ in tqdm(range(N_SAMPLES)):
    example = next(dataset_iter)

    # unsqueeze(0) to add the batch dimension
    input_ids = t.tensor(example["input_ids"]).unsqueeze(0).to(device)
    attention_mask = t.tensor(example["attention_mask"]).unsqueeze(0).to(device)
    total_tokens += input_ids.size(dim=-1)

    with t.no_grad():
        output = model(input_ids, attention_mask, output_router_logits=True)

    # router_logits has shape (`num_layers`, `seq_len`, `num_experts`)
    router_logits = t.stack(output.router_logits)
    router_weights = F.softmax(router_logits, dim=-1)

    # Pick the top k=8 experts
    # Shape: (`num_layers`, `seq_len`, `k`)
    router_weights, expert_ids = t.topk(router_weights, k)

    # Scatter-add: Index into total_weights and add
    # Scatter-add doesn't cast automatically, so cast to float32
    router_weights = router_weights.view(num_layers, -1).float()
    expert_ids = expert_ids.view(num_layers, -1)
    total_weights.scatter_add_(dim=1, index=expert_ids, src=router_weights)

# %% Calculate mean layer weight
# OLMoE does not normalize top-k probability, so the total weight of activated experts is not 1.
# We find the mean total weight of each layer to later preserve each layer's output magnitude.
layer_weight = t.sum(total_weights, dim=-1, keepdim=True) / total_tokens

# Should be close to but not 1
print(layer_weight)

# %% For each layer, find the k experts with the largest total weight
# Then calculate how much each expert's weights should be scaled
extract_weights, extract_ids = t.topk(total_weights, k)
extract_weights *= layer_weight / t.sum(extract_weights, dim=-1, keepdim=True)

# %% Print experts to extract and their scaling factor
print("Expert IDs:")
print(extract_ids)
print()

print("Scaling factor:")
print(extract_weights)
print()
