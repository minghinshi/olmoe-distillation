# %% Setup
import onnxruntime as ort
from transformers import OlmoeForCausalLM
from transformers.exporters.configs import OnnxConfig
from transformers.exporters.exporter_onnx import OnnxExporter

import utils

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

# %% Use Hugging Face to save the model as ONNX
# https://huggingface.co/docs/transformers/exporters
tokenizer = utils.load_tokenizer()
inputs = tokenizer("Hello, world!", return_tensors="pt")

exporter = OnnxExporter()
config = OnnxConfig(dynamic=True)
onnx_program = exporter.export(merged_model, inputs, config=config)

# %% Save the model
onnx_program.save(ONNX_MODEL_PATH)

# %% Test drive the model
session = ort.InferenceSession(ONNX_MODEL_PATH)
ort_inputs = {k: v.numpy() for k, v in inputs.items()}
outputs = session.run(None, ort_inputs)
