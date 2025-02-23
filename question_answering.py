from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load pre-trained QA model and tokenizer
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # SQuAD model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Context and question
context = "Hugging Face is a company based in New York City. It provides tools for natural language processing."
question = "Where is Hugging Face based?"

# Tokenize input
inputs = tokenizer(question, context, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get start and end scores
start_scores = outputs.start_logits
end_scores = outputs.end_logits

# Get the most likely start and end positions
start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores) + 1  # Add 1 to include the end token

# Decode the answer
answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
)
print(f"Question: {question}")
print(f"Answer: {answer}")