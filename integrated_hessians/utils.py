import torch
from transformers import AutoTokenizer

def tokenize_texts(texts, tokenizer_name='roberta-base', device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    return enc['input_ids'].to(device), enc['attention_mask'].to(device)