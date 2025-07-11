# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
import yaml
import os
import random
import numpy as np
from tqdm import tqdm
import logging
import argparse
from pathlib import Path

from model import TextJEPA
from data_utils import get_tokenizer, TextJEPADataset, create_collate_fn

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, dataloader, optimizer, scaler, device, config):
    """The training loop for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        context_tokens = batch['context_tokens'].to(device, non_blocking=True)
        target_tokens = batch['target_tokens'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config['training']['use_amp']):
            output = model(context_tokens, target_tokens, mask)
            loss = output['loss']

        if torch.isnan(loss):
            logging.warning("NaN loss detected. Skipping batch.")
            continue

        scaler.scale(loss).backward()
        
        if config['training']['gradient_clipping'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping'])

        scaler.step(optimizer)
        scaler.update()
        
        model.update_target_encoder()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device, config):
    """The validation loop."""
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)

    for batch in progress_bar:
        context_tokens = batch['context_tokens'].to(device, non_blocking=True)
        target_tokens = batch['target_tokens'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config['training']['use_amp']):
            output = model(context_tokens, target_tokens, mask)
            loss = output['loss']

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def main(args):
    # --- Load Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['system']['seed'])
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Prepare Data ---
    logging.info("Preparing tokenizer and dataset...")
    tokenizer = get_tokenizer(config)
    
    dataset_file = next(Path(config['data']['path']).glob("*.txt"), None)
    if not dataset_file:
        raise FileNotFoundError(f"No .txt file found in {config['data']['path']}")

    dataset = TextJEPADataset(str(dataset_file), tokenizer, config)
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    collate_fn = create_collate_fn(tokenizer.token_to_id("[PAD]"))
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    logging.info(f"Data loaded: {len(train_dataset)} train samples, {len(val_dataset)} validation samples.")

    # --- Initialize Model ---
    logging.info("Initializing Text-JEPA model...")
    config['data']['vocab_size'] = tokenizer.get_vocab_size()
    model = TextJEPA(config).to(device)

    # --- Setup Optimizer, Scheduler, and Scaler ---
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    # Warmup and Cosine Annealing Scheduler
    warmup_epochs = config['training']['warmup_epochs']
    total_epochs = config['training']['epochs']
    
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(enabled=config['training']['use_amp'])

    # --- Training Loop ---
    best_val_loss = float('inf')
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logging.info("Starting training...")
    for epoch in range(total_epochs):
        logging.info(f"--- Epoch {epoch+1}/{total_epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, config)
        val_loss = validate(model, val_loader, device, config)
        
        logging.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, config['logging']['best_model_name'])
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
            logging.info(f"Validation loss improved. Model saved to {checkpoint_path}")

    logging.info("Training finished.")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Text-JEPA model.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    main(args)
