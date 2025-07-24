# data_utils.py - Fixed version
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
from pathlib import Path
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_tokenizer(config: dict):
    """
    Loads a pre-trained BPE tokenizer or trains a new one if it doesn't exist.
    """
    tokenizer_path = config.get("tokenizer_path", "tokenizer.json")
    data_path = config['data']['path']
    vocab_size = config['data']['vocab_size']

    if os.path.exists(tokenizer_path):
        logging.info(f"Loading existing tokenizer from {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    logging.warning(f"Tokenizer not found at {tokenizer_path}. Training a new one from scratch.")
    
    # Handle both single file and directory
    if os.path.isfile(data_path):
        files = [data_path]
    else:
        files = [str(p) for p in Path(data_path).glob("**/*.txt") if p.is_file()]
    
    if not files:
        raise ValueError(f"No .txt files found in {data_path}. Cannot train tokenizer.")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    special_tokens = ["[UNK]", "[PAD]", "[MASK]", "[CLS]", "[SEP]"]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    tokenizer.train(files, trainer)
    tokenizer.save(tokenizer_path)
    logging.info(f"Tokenizer trained and saved to {tokenizer_path}")
    return tokenizer

def create_block_mask(seq_len: int, max_length: int, mask_ratio: float, block_size: int):
    """
    Creates a boolean mask using a block-wise strategy.
    """
    mask = torch.zeros(max_length, dtype=torch.bool)
    num_tokens_to_mask = int(seq_len * mask_ratio)

    if num_tokens_to_mask == 0:
        return mask

    block_size = min(block_size, seq_len)
    
    masked_count = 0
    attempts = 0
    max_attempts = 50

    while masked_count < num_tokens_to_mask and attempts < max_attempts:
        start_idx = random.randint(0, seq_len - block_size)
        end_idx = min(start_idx + block_size, seq_len)
        
        if not mask[start_idx:end_idx].all():
            mask[start_idx:end_idx] = True
            masked_count = mask[:seq_len].sum().item()
        attempts += 1
        
    return mask

class TextJEPADataset(Dataset):
    """
    Dataset for Text-JEPA with proper handling of file paths.
    """
    def __init__(self, file_path: str, tokenizer: Tokenizer, config: dict):
        self.tokenizer = tokenizer
        self.config = config['data']
        self.max_length = self.config['max_length']
        self.mask_ratio = self.config['mask_ratio']
        self.mask_block_size = self.config['mask_block_size']
        
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        
        if self.pad_token_id is None or self.mask_token_id is None:
            raise ValueError("[PAD] or [MASK] token not found in the tokenizer vocabulary.")

        # Handle both single file and directory
        if os.path.isfile(file_path):
            files = [file_path]
        else:
            files = [str(p) for p in Path(file_path).glob("**/*.txt") if p.is_file()]
        
        if not files:
            raise FileNotFoundError(f"No .txt files found in {file_path}")
        
        self.lines = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                self.lines.extend([line.strip() for line in f if line.strip()])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        
        encoding = self.tokenizer.encode(line)
        tokens = encoding.ids
        
        tokens = tokens[:self.max_length]
        seq_len = len(tokens)
        
        target_tokens = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
        target_tokens[:seq_len] = torch.tensor(tokens, dtype=torch.long)
        
        mask = create_block_mask(seq_len, self.max_length, self.mask_ratio, self.mask_block_size)
        
        context_tokens = target_tokens.clone()
        context_tokens[mask] = self.mask_token_id
        
        return {
            "context_tokens": context_tokens,
            "target_tokens": target_tokens,
            "mask": mask
        }

def create_collate_fn(pad_token_id: int):
    """
    Creates a collate function - Fixed version.
    """
    def collate_fn(batch):
        context_tokens = torch.stack([item['context_tokens'] for item in batch])
        target_tokens = torch.stack([item['target_tokens'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        
        return {
            "context_tokens": context_tokens,
            "target_tokens": target_tokens,
            "mask": masks
        }
        
    return collate_fn