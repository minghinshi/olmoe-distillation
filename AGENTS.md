# OLMoE distillation

This project aims to distill OLMoE, a mixture-of-experts model, into a dense version of itself to be used in edge devices.

## Project steps

1. Create a dense version of OLMoE by replacing expert blocks with dense MLP blocks, then transferring weights.
2. Use knowledge distillation + low-rank adaptation with the original OLMoE as the teacher to train the dense OLMoE.
3. Export the dense OLMoE in ONNX format.
4. Evaluate the original and dense OLMoE on benchmarks and compare.

Status: Steps up to step 3 are completed, but the dense model fails to produce a coherent response.

## Targets

- The ONNX model runs on CPU at 5 tokens/s.
- The dense OLMoE achieves 85% of the original OLMoE's score on all evaluated benchmarks.

## Instructions for AI agents

- I am still learning how machine learning works, via this project. Therefore:
    - Please do not modify any file in this project. Let me write code on my own.
    - Please do not provide code blocks (triple ticks) in your responses. Inline code is fine.
- The list of installed packages are at `pyproject.toml` and `requirements.txt`.
    - Please do not download packages. If you want to suggest a package to me, tell me in your response.
