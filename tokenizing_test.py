from transformers import AutoTokenizer

# Load the tokenizer for WangchanBERTa
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# A sample sentence from your dataset
sample_text = "hello world"

# Tokenize the text
encoded_output = tokenizer(sample_text)

# Print the output
print(encoded_output)