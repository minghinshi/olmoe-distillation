# %% Setup
import einops as e
import torch as t
from transformers.models.olmoe.modeling_olmoe import OlmoeMLP, OlmoeSparseMoeBlock

import utils

utils.load_and_check_env()

# Most frequently used experts from analysis
EXPERT_IDS = t.tensor(
    [
        [41, 58, 36, 0, 40, 8, 54, 49],
        [47, 18, 21, 11, 23, 27, 35, 12],
        [60, 45, 14, 34, 22, 11, 15, 23],
        [31, 9, 42, 52, 30, 19, 45, 59],
        [21, 17, 6, 62, 25, 34, 59, 23],
        [0, 21, 42, 33, 31, 52, 17, 22],
        [18, 57, 62, 24, 36, 52, 61, 1],
        [4, 35, 17, 2, 48, 53, 23, 56],
        [54, 16, 37, 27, 45, 11, 57, 10],
        [5, 51, 13, 8, 50, 7, 52, 46],
        [56, 43, 11, 62, 13, 59, 37, 2],
        [47, 27, 57, 44, 54, 33, 23, 43],
        [59, 55, 43, 2, 30, 38, 54, 6],
        [2, 32, 27, 55, 5, 62, 25, 41],
        [9, 58, 52, 60, 11, 1, 63, 7],
        [17, 1, 44, 36, 24, 5, 8, 34],
    ]
)

# How much to scale each expert's down_proj
SCALING = t.tensor(
    [
        [0.0919, 0.0397, 0.0388, 0.0384, 0.0373, 0.0357, 0.0347, 0.0337],
        [0.0520, 0.0474, 0.0423, 0.0334, 0.0329, 0.0297, 0.0293, 0.0278],
        [0.0605, 0.0431, 0.0381, 0.0345, 0.0336, 0.0335, 0.0332, 0.0316],
        [0.0498, 0.0451, 0.0447, 0.0427, 0.0403, 0.0388, 0.0378, 0.0357],
        [0.0583, 0.0513, 0.0427, 0.0376, 0.0314, 0.0307, 0.0298, 0.0295],
        [0.0807, 0.0367, 0.0357, 0.0341, 0.0336, 0.0334, 0.0331, 0.0326],
        [0.0716, 0.0543, 0.0422, 0.0405, 0.0354, 0.0323, 0.0305, 0.0302],
        [0.0622, 0.0523, 0.0482, 0.0429, 0.0397, 0.0383, 0.0376, 0.0345],
        [0.0762, 0.0537, 0.0504, 0.0444, 0.0426, 0.0410, 0.0397, 0.0390],
        [0.0945, 0.0627, 0.0480, 0.0464, 0.0460, 0.0448, 0.0428, 0.0424],
        [0.0791, 0.0684, 0.0605, 0.0584, 0.0557, 0.0485, 0.0470, 0.0459],
        [0.1031, 0.0702, 0.0588, 0.0499, 0.0484, 0.0481, 0.0452, 0.0448],
        [0.0848, 0.0727, 0.0709, 0.0618, 0.0592, 0.0559, 0.0505, 0.0490],
        [0.0930, 0.0735, 0.0679, 0.0665, 0.0613, 0.0536, 0.0471, 0.0451],
        [0.0932, 0.0871, 0.0667, 0.0516, 0.0499, 0.0492, 0.0491, 0.0478],
        [0.0867, 0.0687, 0.0679, 0.0547, 0.0493, 0.0491, 0.0482, 0.0473],
    ]
)

# %% Install the model and load on the CPU
sparse_model = utils.load_olmoe()
print(sparse_model)

# %% Load a dense model with MLP weights not set
dense_model = utils.create_dense_model()
print(dense_model)

# %% Load weights of matching modules from MoE to dense model
dense_model.load_state_dict(sparse_model.state_dict(), strict=False)

# %% Extract experts
for i in range(len(sparse_model.model.layers)):
    sparse_mlp = sparse_model.model.layers[i].mlp
    assert isinstance(sparse_mlp, OlmoeSparseMoeBlock)
    experts = sparse_mlp.experts

    # Shape: (num_experts, intermediate_dim * 2, hidden_dim)
    gate_up_proj = experts.gate_up_proj
    assert isinstance(gate_up_proj, t.Tensor)

    # Shape: (num_experts, hidden_dim, intermediate_dim)
    down_proj = experts.down_proj
    assert isinstance(down_proj, t.Tensor)

    # Index the experts that we want
    gate_up_proj = gate_up_proj[EXPERT_IDS[i]]
    down_proj = down_proj[EXPERT_IDS[i]]

    # Split gate_up_proj into gate_proj, up_proj
    # Shape: (k, intermediate_dim, hidden_dim)
    gate_proj, up_proj = t.chunk(gate_up_proj, 2, dim=1)

    # Scale down down_proj to compensate for router weight
    # (Don't use an in-place operation, just in case it updates teacher weights)
    down_proj = down_proj * SCALING[i].unsqueeze(-1).unsqueeze(-1)

    # Stack the tensors into big 2D matrices
    gate_proj = e.rearrange(gate_proj, "k int hid -> (k int) hid")
    up_proj = e.rearrange(up_proj, "k int hid -> (k int) hid")
    down_proj = e.rearrange(down_proj, "k hid int -> hid (k int)")

    # Transfer weights to the dense model's MLP
    dense_mlp = dense_model.model.layers[i].mlp
    assert isinstance(dense_mlp, OlmoeMLP)

    with t.no_grad():
        dense_mlp.gate_proj.weight.copy_(gate_proj)
        dense_mlp.up_proj.weight.copy_(up_proj)
        dense_mlp.down_proj.weight.copy_(down_proj)

# %% Save the model for the next file
dense_model.save_pretrained(utils.DENSE_MODEL_DIR)

# %% Count parameters
print("Parameter count:")
print(f"{sum(p.numel() for p in dense_model.parameters()):,}")
