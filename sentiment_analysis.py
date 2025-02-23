from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Sentiment analysis model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input text
text = "I love using Hugging Face models!"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
probs = F.softmax(outputs.logits, dim=-1)
predicted_class = torch.argmax(probs, dim=-1).item()

# Get label (e.g., POSITIVE/NEGATIVE)
labels = ["NEGATIVE", "POSITIVE"]
print(f"Input: {text}")
print(f"Predicted class: {labels[predicted_class]} with probability: {probs[0][predicted_class]:.4f}")