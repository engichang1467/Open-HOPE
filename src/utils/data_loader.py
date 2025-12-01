import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len, max_length):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_length = max_length
        # Tokenize all texts
        # In a real scenario, we might stream or map this.
        # For wikitext-103, it's large, so we should be careful.
        # We'll just take a subset or process on the fly for this reimplementation demo.
        
        # self.encodings = tokenizer("\n\n".join(texts), return_tensors='pt', truncation=False, max_length=None)["input_ids"][0]
        # Handle empty texts list
        if not texts:
            self.encodings = torch.tensor([])
        else:
            self.encodings = tokenizer("\n\n".join(texts), return_tensors='pt', truncation=False, max_length=self.max_length)["input_ids"][0]
        
    def __len__(self):
        # return (len(self.encodings) - 1) // self.max_seq_len
        # Ensure we never return negative length
        if len(self.encodings) == 0:
            return 0
        return max(0, (len(self.encodings) - 1) // self.max_seq_len)

    def __getitem__(self, idx):
        if len(self) == 0:
            raise IndexError("Dataset is empty")
            
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len
        # Inputs and targets (shifted by 1)
        # Actually for LM, we usually return input_ids and labels=input_ids
        # The model/loss handles shifting.
        chunk = self.encodings[start_idx:end_idx]
        return chunk

def get_data_loader(dataset_name, batch_size, max_seq_len, split='train', limit=1024):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    if dataset_name == 'wikitext-103':
        dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
    else:
        # Fallback or other datasets
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=split)
        
    # Filter empty texts
    texts = [x['text'] for x in dataset if x['text'].strip()]
    
    # For demo purposes, limit size if it's too huge or just process it
    # Wikitext-103 is ~100MB text.
    
    ds = TextDataset(texts, tokenizer, max_seq_len, limit)
    
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'), drop_last=True)
