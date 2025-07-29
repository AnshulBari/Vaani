import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
import numpy as np
import time
import math
from model import SmallLLM
from tokenizer import SimpleTokenizer

class LLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Process all texts into training sequences
        print("Processing texts into training sequences...")
        self.sequences = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenizer.encode(text)
            
            # Create overlapping sequences for better training
            if len(tokens) > max_length:
                # Split long texts into overlapping chunks
                stride = max_length // 2
                for i in range(0, len(tokens) - max_length + 1, stride):
                    sequence = tokens[i:i + max_length]
                    self.sequences.append(sequence)
            else:
                # Pad shorter sequences
                padded = tokens + [tokenizer.special_tokens['<pad>']] * (max_length - len(tokens))
                self.sequences.append(padded[:max_length])
        
        print(f"Created {len(self.sequences)} training sequences")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence, dtype=torch.long)

def create_sample_dataset():
    """Create a comprehensive sample dataset for training"""
    texts = [
        # Technology and AI
        "Artificial intelligence represents one of the most significant technological advances in human history. Machine learning algorithms can process vast amounts of data to identify patterns and make predictions.",
        "Deep learning models use neural networks with multiple layers to learn complex representations of data. These models have achieved remarkable success in computer vision, natural language processing, and speech recognition.",
        "The transformer architecture revolutionized natural language processing by introducing the attention mechanism. This allows models to focus on relevant parts of the input sequence when making predictions.",
        
        # Science and Nature
        "The universe contains billions of galaxies, each with billions of stars. Scientists study these cosmic structures to understand the fundamental laws of physics and the origins of matter and energy.",
        "Climate change is affecting ecosystems worldwide. Rising temperatures, changing precipitation patterns, and extreme weather events are reshaping the natural world in unprecedented ways.",
        "Quantum mechanics describes the behavior of matter and energy at the smallest scales. This field has led to revolutionary technologies including lasers, transistors, and quantum computers.",
        
        # Literature and Philosophy
        "Literature serves as a mirror to society, reflecting the values, struggles, and aspirations of different cultures and time periods. Through stories, we explore the human condition and gain insights into ourselves.",
        "Philosophy examines fundamental questions about existence, knowledge, ethics, and reality. These inquiries have shaped human thought for millennia and continue to influence how we understand our place in the world.",
        "Critical thinking involves analyzing information objectively and making reasoned judgments. This skill is essential for navigating complex problems and making informed decisions in all areas of life.",
        
        # History and Culture
        "Historical events shape the present in ways that are often invisible to contemporary observers. Understanding the past helps us recognize patterns and make better decisions for the future.",
        "Cultural diversity enriches human experience through different perspectives, traditions, and ways of understanding the world. Preserving and celebrating this diversity is crucial for a thriving global society.",
        "Innovation often emerges from the intersection of different fields and disciplines. Breakthrough discoveries frequently occur when experts from various domains collaborate and share their unique insights.",
        
        # Mathematics and Logic
        "Mathematical concepts provide the foundation for understanding patterns in nature and developing technological solutions. From basic arithmetic to advanced calculus, mathematics is the language of science.",
        "Logical reasoning enables us to draw valid conclusions from given premises. This systematic approach to thinking is fundamental to scientific inquiry, problem-solving, and decision-making.",
        "Statistics help us make sense of data and uncertainty in the real world. By analyzing patterns and trends, we can make informed predictions and understand the likelihood of different outcomes.",
    ]
    
    # Expand the dataset by repeating and slightly modifying texts
    expanded_texts = []
    for text in texts:
        expanded_texts.append(text)
        # Add variations
        sentences = text.split('. ')
        if len(sentences) > 1:
            # Create variations by reordering sentences
            for i in range(min(3, len(sentences))):
                if i < len(sentences) - 1:
                    reordered = sentences[i+1:] + sentences[:i+1]
                    expanded_texts.append('. '.join(reordered))
    
    # Multiply dataset for more training data
    final_texts = expanded_texts * 20  # 20x repetition for substantial training
    
    print(f"Created dataset with {len(final_texts)} text samples")
    return final_texts

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def train_model():
    # Configuration optimized for your model
    config = {
        'vocab_size': 15000,    # Reasonable vocabulary size
        'd_model': 1024,        # Model dimension for ~4GB target
        'n_layers': 16,         # Number of transformer layers
        'n_heads': 16,          # Attention heads
        'd_ff': 4096,           # Feed-forward dimension
        'max_seq_len': 512,     # Sequence length
        'dropout': 0.1,
        
        # Training parameters
        'batch_size': 4,
        'gradient_accumulation_steps': 8,  # Effective batch size = 4 * 8 = 32
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'epochs': 3,
        'warmup_ratio': 0.1,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=== Training Configuration ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 30)
    
    # Create and prepare dataset
    texts = create_sample_dataset()
    
    # Initialize tokenizer
    print("Training tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
    tokenizer.train(texts)
    tokenizer.save('tokenizer.pkl')
    print(f"Tokenizer vocabulary size: {len(tokenizer.word_to_id)}")
    
    # Create dataset and dataloader
    dataset = LLMDataset(texts, tokenizer, max_length=config['max_seq_len'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Initialize model with your architecture
    model = SmallLLM(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    print(f"\n=== Model Information ===")
    print(f"Total parameters: {model.count_parameters():,}")
    print(f"Estimated model size: {model.count_parameters() * 4 / (1024**3):.2f} GB (FP32)")
    print(f"Model device: {config['device']}")
    print("=" * 30)
    
    model = model.to(config['device'])
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * config['epochs'] // config['gradient_accumulation_steps']
    warmup_steps = int(config['warmup_ratio'] * total_steps)
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=config['learning_rate'],
        min_lr=config['learning_rate'] * 0.1
    )
    
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['epochs']} ===")
        
        total_loss = 0
        optimizer.zero_grad()
        
        epoch_start_time = time.time()
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(config['device'])
            
            # Forward pass using your model's forward method
            loss, logits = model(batch, labels=batch)
            
            # Scale loss for gradient accumulation
            loss = loss / config['gradient_accumulation_steps']
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * config['gradient_accumulation_steps']
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
                
                # Learning rate scheduling
                current_lr = scheduler.step()
                global_step += 1
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * config["gradient_accumulation_steps"]:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': global_step
                })
                
                # Save checkpoint every 500 steps
                if global_step % 500 == 0:
                    checkpoint_path = f'checkpoint_step_{global_step}.pth'
                    torch.save({
                        'step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'config': config
                    }, checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
        print(f"Average loss: {avg_epoch_loss:.4f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Loss: {best_loss:.4f}")
        
        # Save epoch checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'config': config
        }, f'checkpoint_epoch_{epoch + 1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    
    # Save configuration
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n=== Training Completed! ===")
    print(f"Best loss achieved: {best_loss:.4f}")
    print(f"Final model saved as: final_model.pth")
    print(f"Best model saved as: best_model.pth")
    print(f"Tokenizer saved as: tokenizer.pkl")
    print("=" * 30)
    
    return model, tokenizer, config

if __name__ == "__main__":
    model, tokenizer, config = train_model()
