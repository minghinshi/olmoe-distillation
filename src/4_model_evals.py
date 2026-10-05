# %% Setup
import utils

# %% Load your Hugging Face token
utils.load_and_check_env()

# %% Load the three models to evaluate and the tokenizer
original_model = utils.load_quantized_moe()
dense_model = utils.load_dense_model()
distilled_model = utils.load_distilled_model()
tokenizer = utils.load_tokenizer()

# %% Test drive all three models
# Make sure the distilled model isn't producing gibberish
print("Original model:")
utils.test_drive(original_model, tokenizer)

print("Dense model:")
utils.test_drive(dense_model, tokenizer)

print("Distilled model:")
utils.test_drive(distilled_model, tokenizer)
