from torch import nn
from transformers import OlmoeConfig, OlmoeForCausalLM, OlmoeModel
from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer, OlmoeMLP


class DenseOlmoeConfig(OlmoeConfig):
    model_type = "dense_olmoe"


class DenseOlmoeDecoderLayer(OlmoeDecoderLayer):
    def __init__(self, config: DenseOlmoeConfig, layer_idx: int):
        super().__init__(config, layer_idx)

        del self.mlp
        self.mlp = OlmoeMLP(config)


class DenseOlmoeModel(OlmoeModel):
    def __init__(self, config: DenseOlmoeConfig):
        super().__init__(config)

        del self.layers
        self.layers = nn.ModuleList(
            [DenseOlmoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


class DenseOlmoeForCausalLM(OlmoeForCausalLM):
    config_class = DenseOlmoeConfig

    def __init__(self, config: DenseOlmoeConfig):
        super().__init__(config)

        del self.model
        self.model = DenseOlmoeModel(config)
