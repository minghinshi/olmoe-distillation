# OLMoE distillation

## Methods

### Hardware and software requirements

This project used an NVIDIA RTX A4000 rented from vast.ai. The A4000 has 16 GB VRAM, which is enough for inference of a ~7B model at 16-bit precision. Vast.ai containers supporting Python 3.12 and CUDA 13.0 were used. About 30 GB disk space is required.

### Model

The model used was an instruction fine-tuned OLMoE, available on Hugging Face as `allenai/OLMoE-1B-7B-0924-Instruct`. OLMoE is an open-weight, open-source mixture-of-experts (MoE) model, with 7B total parameters and 1B active parameters. It consists of 16 layers of attention and MLP. Each MLP block has 64 experts, 8 of which are activated per token. Experts use the SwiGLU activation function. Tokens have dimension 2048, while experts have intermediate dimension 1024.

Larger and more popular MoE models such as `gpt-oss-20b` (21B parameters) or `Qwen1.5-MoE-A2.7B` (14B parameters) may fit in memory with optimization such as 4-bit quantization. However, this is the first time I trained an LLM, so I wanted more margin for error.

### MoE quantization

Directly loading OLMoE in VRAM consumes 14 GB out of 16 GB. The remaining memory is insufficient for distillation. The `bitsandbytes` library offers 4-bit quantization. However, it did not work well with OLMoE, because it quantizes `nn.Linear` modules, whereas OLMoE's uses `F.linear`. To address this, OLMoE’s experts were replaced with a custom version using the Linear module. This reduces consumption to 7 GB on load.

### Expert extraction

Three methods were attempted:

1. Extract the first expert from each layer.
2. Average the weights of all experts.
3. Pass 100 examples from a dataset through OLMoE and record the mean router weight of each expert. Extract the expert with the largest mean router weight in each layer.

The repository currently contains code for the last method.

To produce the student model, each layer’s experts were replaced with an MLP whose configuration is identical to an expert (2048 input dimension, 1024 intermediate dimension), then the expert weights are copied to the MLP.

The student model has 575M parameters, less than 10% of the teacher’s.

### LoRA distillation

Low-rank adaptation configuration mostly followed standard practice: rank 16, alpha 32, dropout 0, batch size 2, gradient accumulation steps 8, adapters target both attention and MLP parameters. Batch size 4 did not fit in memory. There were about 6M trainable paramters, which was 1% of total parameters. This seemed appropriate, so rank was not further increased to 32.

The dataset used was the OLMoE supervised fine-tuning (SFT) mix (Hugging Face: `allenai/tulu-v3.1-mix-preview-4096-OLMoE`). A previous training run used the Colossal Clean Crawled Corpus (C4), but an SFT dataset is more appropriate because we want the student to be able to chat.

Loss was defined as the KL divergence of the student distribution from the teacher distribution. A temperature of 2 was tentatively used for both distributions.

### Training runs

Two training runs were conducted. In the first training run, the first expert was extracted, the C4 dataset was used, and LoRA only targeted attention. 4,000 training steps were performed in about 16 hours.

After the first run, the student was unable to chat, so a second training run was conducted using methods described above. Because the first run's loss plateaued early, the number of training steps was reduced to 2,000. The second run took 9 hours.

## Results and discussion

The student model before distillation, distillation checkpoints, and the final model in ONNX format are available on Hugging Face at `minghinshi/OLMoE-distill`.

The first run's final loss was about 40. The second run's loss reached 40 in about 700 steps, 5.7x the efficiency of the first run. The second run's final loss was about 34.

Both run's final models were unable to chat. When prompted with "Who are you?", the first run's model mostly output new line characters. The second run's promisingly started with "The answer is:", but then output common words like "the", "a", and "is".

Results of running `src/4_model_evals.py`:

```
Original model:
Hello! I am an AI language model designed to assist with a wide range of tasks, including answering questions, providing information, and much more. My primary function is to help people find the answers they

Dense model:
VOLinse. \oud- $alin          (`ree($\HL approximation\_ trainers(ops $ifen-ookie (` ( hän $ $ $ Lips forth - '\Alreadywood\itetbsite- -(

Distilled model:
The answer is: "The "The" of the first "A" is a "A" and a "A" is a "A" and a "A" is a "A"
```

This suggests that distillation was working but needs to be optimized further before producing a usable model.

## AI usage disclosure

AI models were used for advice and step-by-step instructions. They were not allowed to generate code blocks because these can be copied and pasted, but they could use lines of code in their explanations. No coding agents were used.
