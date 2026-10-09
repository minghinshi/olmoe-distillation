# %% Setup
import onnxruntime as ort
import torch as t
from transformers import OlmoeForCausalLM
from transformers.exporters.configs import OnnxConfig
from transformers.exporters.exporter_onnx import OnnxExporter

import utils

utils.load_and_check_env()

ONNX_MODEL_PATH = "../models/onnx_model/model.onnx"

# %% Load the distilled model
model = utils.load_distilled_model()
print(model)

# %% Configure the model for a clean export

# Simplify the graph by merging LoRA weights into the model
merged_model: OlmoeForCausalLM = model.merge_and_unload()

# Switch to eval mode for dropout, etc. to work properly
merged_model.eval()

# ONNX might work better on CPU
merged_model.to("cpu")  # type: ignore

# The CPU execution provider doesn't support `bfloat16` for the operation `Mul(14)`.
# We need to convert the model to `float32`, but this doubles the size of the model.
merged_model.to(t.float32)  # type: ignore

# %% Prepare example input
tokenizer = utils.load_tokenizer()
inputs = tokenizer("Hello, world!", return_tensors="pt")

# %% Use Hugging Face to save the model as ONNX
# https://huggingface.co/docs/transformers/exporters
exporter = OnnxExporter()
config = OnnxConfig(dynamic=True)
onnx_program = exporter.export(merged_model, inputs, config=config)

# %% Save the model
onnx_program.save(ONNX_MODEL_PATH)

# %% Test drive the model
session = ort.InferenceSession(ONNX_MODEL_PATH)
ort_inputs = {k: v.numpy() for k, v in inputs.items()}
outputs = session.run(None, ort_inputs)

onnx_logits = t.tensor(outputs[0])
pytorch_logits = merged_model(**inputs).logits
assert t.allclose(onnx_logits, pytorch_logits, atol=1e-4)
