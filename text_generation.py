from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input prompt
prompt = "Once upon a time there was a squirrel"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    inputs["input_ids"],
    max_length=500,  # Maximum length of generated text
    num_return_sequences=1,  # Number of sequences to generate
    no_repeat_ngram_size=2,  # Prevent repetition
    do_sample=True,  # Enable sampling
    top_k=50,  # Top-k sampling
    top_p=0.95,  # Nucleus sampling
)

# Decode generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")