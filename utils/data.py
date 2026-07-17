import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import itertools

def get_tokenizer_path(tokenizer_name):
    name_lower = tokenizer_name.lower()
    if name_lower == "gpt2":
        return "gpt2"
    elif name_lower == "qwen":
        return "Qwen/Qwen2.5-0.5B"
    return tokenizer_name

class HuggingFaceDataset(Dataset):
    """
    A PyTorch Dataset that loads a text dataset from Hugging Face,
    tokenizes it using a pretrained tokenizer, and chunks it into
    sequences of seq_len for causal language modeling.
    Supports WikiText, TinyStories, and FineWeb.
    """
    def __init__(self, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="train", tokenizer_name="gpt2", seq_len=128, max_samples=None):
        name_lower = dataset_name.lower()
        if "wikitext" in name_lower:
            dataset_name = "wikitext"
            if not dataset_config:
                dataset_config = "wikitext-2-raw-v1"
        elif "tinystories" in name_lower:
            dataset_name = "roneneldan/TinyStories"
            dataset_config = None
        elif "fineweb" in name_lower:
            dataset_name = "HuggingFaceFW/fineweb"
            if not dataset_config:
                dataset_config = "sample-100M"

        print(f"Loading dataset '{dataset_name}' (config: {dataset_config}) split '{split}' from Hugging Face...")
        
        try:
            if dataset_config:
                raw_dataset = load_dataset(dataset_name, dataset_config, split=split)
            else:
                raw_dataset = load_dataset(dataset_name, split=split)
        except ValueError as e:
            if split in ["validation", "val"]:
                print(f"Validation split not found. Loading train split to slice validation set...")
                if dataset_config:
                    full_dataset = load_dataset(dataset_name, dataset_config, split="train")
                else:
                    full_dataset = load_dataset(dataset_name, split="train")
                
                val_size = int(len(full_dataset) * 0.1)
                if max_samples:
                    val_size = min(val_size, max_samples)
                raw_dataset = full_dataset.select(range(len(full_dataset) - val_size, len(full_dataset)))
            else:
                raise e
        
        if max_samples is not None and max_samples < len(raw_dataset):
            print(f"Slicing dataset to first {max_samples} samples...")
            raw_dataset = raw_dataset.select(range(max_samples))
            
        tokenizer_path = get_tokenizer_path(tokenizer_name)
        print(f"Loading tokenizer '{tokenizer_name}' from path '{tokenizer_path}'...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Tokenizing dataset...")
        text_column = "text"
        if text_column not in raw_dataset.column_names:
            for col in raw_dataset.column_names:
                if col in ["text", "content", "document"]:
                    text_column = col
                    break
        
        texts = [text for text in raw_dataset[text_column] if text and text.strip()]
        
        encodings = self.tokenizer(texts, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id
        
        token_lists = [ids + [eos_id] for ids in encodings["input_ids"]]
        self.tokens = list(itertools.chain.from_iterable(token_lists))
        self.seq_len = seq_len
        
        print(f"Tokenization complete. Total tokens: {len(self.tokens)}")

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        chunk = self.tokens[start_idx:end_idx + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloader(dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", split="train", tokenizer_name="gpt2", seq_len=128, batch_size=8, shuffle=True, max_samples=None):
    dataset = HuggingFaceDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        tokenizer_name=tokenizer_name,
        seq_len=seq_len,
        max_samples=max_samples
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=2
    )
    return loader, dataset.tokenizer

def get_dataloaders(config):
    """
    Convenience function to return train and validation loaders using a ModelConfig instance.
    """
    train_loader, tokenizer = get_dataloader(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split="train",
        tokenizer_name=config.tokenizer_name,
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        shuffle=True,
        max_samples=config.max_samples
    )
    
    val_loader, _ = get_dataloader(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split="validation",
        tokenizer_name=config.tokenizer_name,
        seq_len=config.max_seq_len,
        batch_size=config.batch_size,
        shuffle=False,
        max_samples=config.val_max_samples
    )
    
    return train_loader, val_loader, tokenizer
