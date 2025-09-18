from transformers import AutoTokenizer
import pandas as pd
from datasets import Dataset

# Load the tokenizer for WangchanBERTa
model_name = "airesearch/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # 'text_column_name' should be the name of the column in your dataset with the text
    return tokenizer(
        examples["Message"], 
        padding="max_length", # Pad shorter sentences to the max length
        truncation=True       # Truncate sentences longer than the model can handle
        
    )

df = pd.read_csv('data/HateThaiSent.csv')

# Assume 'df' is your pandas DataFrame with the HateThaiSent data
# 1. Convert your DataFrame to a Dataset object
hg_dataset = Dataset.from_pandas(df)

# 2. Apply the tokenization function to the entire dataset
# The 'batched=True' argument makes the process much faster
tokenized_dataset = hg_dataset.map(tokenize_function, batched=True)

# 3. Check the result
print(tokenized_dataset)