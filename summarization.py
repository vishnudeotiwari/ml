from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"  # Summarization model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input text
text = """
Hugging Face is a company based in New York City. It provides tools for natural language processing, 
including the popular Transformers library. The company was founded in 2016 and has since grown 
to become a leading name in the AI community.
"""

# Tokenize input
inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)

# Generate summary
summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=50, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"Summary: {summary}")