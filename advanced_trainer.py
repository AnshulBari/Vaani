import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import json
import time
import math
from tqdm import tqdm
import numpy as np

from model import SmallLLM
from tokenizer import SimpleTokenizer
from data_sources import create_comprehensive_dataset, get_training_config

class TrainingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, overlap_ratio=0.3):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        
        print("Processing training data...")
        
        # Convert texts to token sequences
        all_tokens = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.special_tokens['<eos>'])  # Separate documents
        
        print(f"Total tokens collected: {len(all_tokens):,}")
        
        # Create overlapping sequences
        stride = int(max_length * (1 - overlap_ratio))
        
        for i in range(0, len(all_tokens) - max_length + 1, stride):
            sequence = all_tokens[i:i + max_length]
            self.sequences.append(sequence)
        
        print(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

class LearningRateScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = peak_lr * min_lr_ratio
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.peak_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Mixed precision training
        self.use_amp = config['mixed_precision'] and config['device'] == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        print(f"Using device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        
        # Initialize model
        self.model = SmallLLM(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config['dropout']
        ).to(self.device)
        
        param_count = self.model.count_parameters()
        model_size_gb = param_count * 4 / (1024**3)
        
        print(f"\n=== Model Information ===")
        print(f"Model parameters: {param_count:,}")
        print(f"Model size: {model_size_gb:.2f} GB (FP32)")
        print("=" * 30)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], config['beta2']),
            weight_decay=config['weight_decay'],
            eps=config['eps']
        )
        
        # Training metrics
        self.train_losses = []
        self.learning_rates = []
        self.best_loss = float('inf')
        self.global_step = 0
        
    def save_checkpoint(self, epoch, step, loss, is_best=False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'learning_rates': self.learning_rates,
            'best_loss': self.best_loss
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = f'checkpoint_step_{step}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, 'best_checkpoint.pth')
            torch.save(self.model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Loss: {loss:.4f}")
        
        return checkpoint_path
    
    def train_step(self, batch):
        """Single training step"""
        batch = batch.to(self.device)
        
        with autocast(enabled=self.use_amp):
            loss, logits = self.model(batch, labels=batch)
            loss = loss / self.config['gradient_accumulation_steps']
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config['gradient_accumulation_steps']
    
    def train_epoch(self, dataloader, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        epoch_start_time = time.time()
        
        for step, batch in enumerate(progress_bar):
            # Training step
            step_loss = self.train_step(batch)
            total_loss += step_loss
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['gradient_clipping']
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate and step counter
                current_lr = scheduler.step()
                self.global_step += 1
                
                # Log metrics
                self.learning_rates.append(current_lr)
                self.train_losses.append(step_loss)
                
                # Update progress bar
                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({
                    'loss': f'{step_loss:.4f}',
                    'avg': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': self.global_step
                })
                
                # Save checkpoint
                if self.global_step % self.config['save_every'] == 0:
                    is_best = avg_loss < self.best_loss
                    if is_best:
                        self.best_loss = avg_loss
                    
                    self.save_checkpoint(epoch, self.global_step, avg_loss, is_best)
        
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / num_batches
        
        print(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        print(f"Global steps completed: {self.global_step}")
        
        return avg_epoch_loss
    
    def train(self, train_texts):
        """Main training loop"""
        print("=== Starting Training ===")
        
        # Prepare tokenizer
        print("Training tokenizer...")
        tokenizer = SimpleTokenizer(vocab_size=self.config['vocab_size'])
        tokenizer.train(train_texts)
        tokenizer.save('tokenizer.pkl')
        print(f"Tokenizer vocabulary size: {len(tokenizer.word_to_id)}")
        
        # Prepare dataset
        dataset = TrainingDataset(
            train_texts, 
            tokenizer, 
            max_length=self.config['max_seq_len'],
            overlap_ratio=self.config['overlap_ratio']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=self.config['shuffle_data'],
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory'] and self.config['device'] == 'cuda'
        )
        
        # Setup scheduler
        steps_per_epoch = len(dataloader) // self.config['gradient_accumulation_steps']
        total_steps = steps_per_epoch * self.config['epochs']
        
        scheduler = LearningRateScheduler(
            self.optimizer,
            warmup_steps=self.config['warmup_steps'],
            total_steps=total_steps,
            peak_lr=self.config['learning_rate']
        )
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {self.config['warmup_steps']}")
        
        # Training loop
        training_start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            print(f"\n=== Epoch {epoch + 1}/{self.config['epochs']} ===")
            
            epoch_loss = self.train_epoch(dataloader, scheduler, epoch)
            
            # Save epoch checkpoint
            self.save_checkpoint(epoch + 1, self.global_step, epoch_loss)
        
        # Final saves
        torch.save(self.model.state_dict(), 'final_model.pth')
        
        # Save training configuration
        with open('model_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training metrics
        with open('training_metrics.json', 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'learning_rates': self.learning_rates,
                'best_loss': self.best_loss,
                'total_steps': self.global_step
            }, f, indent=2)
        
        total_training_time = time.time() - training_start_time
        
        print(f"\n=== Training Completed ===")
        print(f"Total time: {total_training_time / 3600:.2f} hours")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Total steps: {self.global_step}")
        print(f"Final model saved as: final_model.pth")
        print(f"Best model saved as: best_model.pth")
        print(f"Tokenizer saved as: tokenizer.pkl")
        print("=" * 30)

def main():
    # Get configuration
    config = get_training_config()
    
    print("=== Training Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 30)
    
    # Load training data
    print("Creating training dataset...")
    train_texts = create_comprehensive_dataset()
    print(f"Loaded {len(train_texts)} training samples")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Start training
    trainer.train(train_texts)

if __name__ == "__main__":
    main()
