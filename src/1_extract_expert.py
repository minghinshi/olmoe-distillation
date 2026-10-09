# %% Setup
import torch as t

import utils

utils.load_and_check_env()

# Most frequently used experts from analysis
EXPERT_IDS = (41, 18, 60, 9, 21, 0, 57, 4, 54, 5, 56, 47, 59, 2, 58, 17)

# %% Install the model and load on the CPU
sparse_model = utils.load_olmoe()
print(sparse_model)

# %% Load a dense model with MLP weights not set
dense_model = utils.load_dense_model_incomplete()
print(dense_model)

# %% Extract experts
for i in range(len(sparse_model.model.layers)):
    # Get weights from expert 0
    experts = sparse_model.model.layers[i].mlp.experts
    expert_idx = EXPERT_IDS[i]

    gate_up_proj: t.Tensor = experts.gate_up_proj[expert_idx]
    down_proj: t.Tensor = experts.down_proj[expert_idx]
    gate_proj, up_proj = gate_up_proj.chunk(2)

    # Transfer weights to the dense model's MLP
    mlp = dense_model.model.layers[i].mlp

    with t.no_grad():
        mlp.gate_proj.weight.copy_(gate_proj)
        mlp.up_proj.weight.copy_(up_proj)
        mlp.down_proj.weight.copy_(down_proj)

# %% Save the model for the next file
dense_model.save_pretrained(utils.DENSE_MODEL_PATH)
