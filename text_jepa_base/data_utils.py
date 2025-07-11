# data_utils.py
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

# Setup basic logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_tokenizer(config: dict):
    """
    Loads a pre-trained BPE tokenizer or trains a new one if it doesn't exist.
    In a production setup, the tokenizer should be pre-trained and versioned.
    """
    tokenizer_path = config.get("tokenizer_path", "tokenizer.json")
    data_path = config['data']['path']
    vocab_size = config['data']['vocab_size']

    if os.path.exists(tokenizer_path):
        logging.info(f"Loading existing tokenizer from {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    logging.warning(f"Tokenizer not found at {tokenizer_path}. Training a new one from scratch.")
    
    files = [str(p) for p in Path(data_path).glob("**/*.txt") if p.is_file()]
    if not files:
        raise ValueError(f"No .txt files found in {data_path}. Cannot train tokenizer.")

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Special tokens are crucial for many models.
    special_tokens = ["[UNK]", "[PAD]", "[MASK]", "[CLS]", "[SEP]"]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    tokenizer.train(files, trainer)
    tokenizer.save(tokenizer_path)
    logging.info(f"Tokenizer trained and saved to {tokenizer_path}")
    return tokenizer

def create_block_mask(seq_len: int, max_length: int, mask_ratio: float, block_size: int):
    """
    Creates a boolean mask using a block-wise strategy. This version is more robust.
    """
    mask = torch.zeros(max_length, dtype=torch.bool)
    num_tokens_to_mask = int(seq_len * mask_ratio)

    if num_tokens_to_mask == 0:
        return mask

    # Ensure block size is not larger than the sequence length
    block_size = min(block_size, seq_len)
    
    masked_count = 0
    attempts = 0
    max_attempts = 20 # Prevent infinite loops

    while masked_count < num_tokens_to_mask and attempts < max_attempts:
        start_idx = random.randint(0, seq_len - block_size)
        end_idx = start_idx + block_size
        
        # Mask the block if it doesn't increase the masked count too much
        if not mask[start_idx:end_idx].all():
            mask[start_idx:end_idx] = True
            new_masked_count = mask.sum().item()
            # If we overshoot, we can revert, but for simplicity, we allow slight over-masking
            masked_count = new_masked_count
        attempts += 1
        
    return mask

class TextJEPADataset(Dataset):
    """
    Production-ready Dataset for Text-JEPA.
    - Handles large files by reading line-by-line (though still loads all lines).
    - Uses the correct [MASK] token for creating the context.
    - Pads all sequences to a fixed max_length for efficiency.
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

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        # NOTE: For truly massive datasets (e.g., >100GB), loading all lines into memory is
        # not feasible. A better approach is to use a memory-mapped file or a library
        # like `datasets` by Hugging Face which handles streaming efficiently.
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = [line for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        
        encoding = self.tokenizer.encode(line)
        tokens = encoding.ids
        
        # Truncate if longer than max_length
        tokens = tokens[:self.max_length]
        seq_len = len(tokens)
        
        # Create the full-sequence target tensor, padded to max_length
        target_tokens = torch.full((self.max_length,), self.pad_token_id, dtype=torch.long)
        target_tokens[:seq_len] = torch.tensor(tokens, dtype=torch.long)
        
        # Generate the block mask for this sequence
        mask = create_block_mask(seq_len, self.max_length, self.mask_ratio, self.mask_block_size)
        
        # Create context tokens: start with the full sequence and apply the mask
        context_tokens = target_tokens.clone()
        context_tokens[mask] = self.mask_token_id
        
        return {
            "context_tokens": context_tokens,
            "target_tokens": target_tokens,
            "mask": mask
        }

def create_collate_fn(pad_token_id: int):
    """
    Creates a collate function. Since padding is handled in the Dataset, 
    this function just needs to stack the tensors.
    """
    def collate_fn(batch):
        # All items are already padded to max_length, so we can stack them.
        context_tokens = torch.stack([item['context_tokens'] for item in batch])
        target_tokens = torch.stack([item['target_tokens'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        
        return {
            "context_tokens": padded_context,
            "target_tokens": padded_target,
            "mask": padded_masks
        }
        
    return collate_fn
