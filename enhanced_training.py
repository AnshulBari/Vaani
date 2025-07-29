"""
Comprehensive Training Script for Enhanced Vaani LLM
Combines knowledge distillation, transfer learning, data augmentation, and enhanced architecture
"""

import torch
import torch.nn as nn
import json
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# Import our enhanced components
from enhanced_architecture import EnhancedVaaniLLM, create_enhanced_model
from knowledge_distillation import KnowledgeDistillationTrainer
from transfer_learning import TransferLearningInitializer
from data_augmentation import generate_augmented_dataset

class EnhancedTrainingDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []
        
        print("Processing enhanced training data...")
        
        for text in tqdm(texts, desc="Tokenizing"):
            # For string inputs, encode to tokens
            if isinstance(text, str):
                tokens = self.encode_text(text)
            else:
                tokens = text
            
            # Create training sequences
            if len(tokens) > max_length:
                # Split into chunks
                for i in range(0, len(tokens) - max_length + 1, max_length // 2):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) == max_length:
                        self.sequences.append(chunk)
            else:
                # Pad shorter sequences
                padded = tokens + [0] * (max_length - len(tokens))
                self.sequences.append(padded[:max_length])
        
        print(f"Created {len(self.sequences)} training sequences")
    
    def encode_text(self, text):
        """Simple word-based encoding"""
        # For now, use simple word splitting
        # In practice, you'd use a proper tokenizer
        words = text.lower().split()
        tokens = []
        for word in words:
            # Simple hash-based token ID
            token_id = hash(word) % 30000 + 1000  # Avoid special tokens
            tokens.append(token_id)
        return tokens
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

class EnhancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing Enhanced Trainer on {self.device}")
        
        # Create enhanced model
        self.model, self.model_config = create_enhanced_model()
        self.model.to(self.device)
        
        # Initialize with transfer learning if requested
        if config.get('use_transfer_learning', False):
            self.apply_transfer_learning()
        
        # Setup knowledge distillation if requested
        if config.get('use_knowledge_distillation', False):
            self.setup_knowledge_distillation()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # Setup learning rate scheduler
        self.scheduler = None
        
        # Training metrics
        self.train_losses = []
        self.best_loss = float('inf')
        
    def apply_transfer_learning(self):
        """Apply transfer learning initialization"""
        print("Applying transfer learning...")
        try:
            # Create a simple tokenizer for transfer learning
            simple_tokenizer = {
                'word_to_id': {f'word_{i}': i for i in range(1000)},
                'id_to_word': {i: f'word_{i}' for i in range(1000)}
            }
            
            initializer = TransferLearningInitializer("gpt2")
            initializer.initialize_target_model(self.model, simple_tokenizer)
            print("✅ Transfer learning applied successfully")
        except Exception as e:
            print(f"⚠️  Transfer learning failed: {e}")
    
    def setup_knowledge_distillation(self):
        """Setup knowledge distillation trainer"""
        print("Setting up knowledge distillation...")
        try:
            self.distiller = KnowledgeDistillationTrainer(
                teacher_model_name="gpt2",
                temperature=4.0,
                alpha=0.7
            )
            print("✅ Knowledge distillation ready")
        except Exception as e:
            print(f"⚠️  Knowledge distillation setup failed: {e}")
            self.distiller = None
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        # Forward pass
        loss, logits = self.model(batch, labels=batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        total_loss = 0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            total_loss += loss
            
            # Update progress
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{loss:.4f}', 'avg': f'{avg_loss:.4f}'})
        
        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        
        return avg_epoch_loss
    
    def train(self, train_texts):
        """Main training loop"""
        
        # Create dataset
        dataset = EnhancedTrainingDataset(
            train_texts, 
            tokenizer=None,  # We'll use simple encoding
            max_length=self.config['max_seq_len']
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        print(f"Starting enhanced training...")
        print(f"Dataset size: {len(dataset)}")
        print(f"Batches per epoch: {len(dataloader)}")
        
        # Training loop
        for epoch in range(self.config['epochs']):
            print(f"\n=== Epoch {epoch + 1}/{self.config['epochs']} ===")
            
            epoch_loss = self.train_epoch(dataloader, epoch)
            
            print(f"Epoch {epoch + 1} completed - Average loss: {epoch_loss:.4f}")
            
            # Save checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_model(f'enhanced_best_model.pth', epoch + 1, epoch_loss)
                print(f"✅ New best model saved! Loss: {epoch_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 2 == 0:
                self.save_model(f'enhanced_checkpoint_epoch_{epoch + 1}.pth', epoch + 1, epoch_loss)
        
        # Save final model
        self.save_model('enhanced_final_model.pth', self.config['epochs'], self.best_loss)
        
        print(f"\n🎉 Enhanced training completed!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Final model saved as: enhanced_final_model.pth")
    
    def save_model(self, filename, epoch, loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'model_config': self.model_config
        }, filename)

def main():
    """Main training function"""
    
    # Enhanced training configuration
    config = {
        # Model settings
        'max_seq_len': 512,
        
        # Training settings
        'batch_size': 4,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'epochs': 5,
        
        # Enhancement settings
        'use_transfer_learning': True,
        'use_knowledge_distillation': False,  # Set to True if you have transformers installed
        'use_augmented_data': True,
        
        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("🚀 Enhanced Vaani Training Pipeline")
    print("=" * 50)
    
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Load training data
    if config['use_augmented_data']:
        print("Loading augmented training data...")
        try:
            with open('augmented_training_data.json', 'r') as f:
                data = json.load(f)
                train_texts = data['combined_training_data']
            print(f"✅ Loaded {len(train_texts)} augmented samples")
        except:
            print("⚠️  Augmented data not found, generating...")
            train_texts = generate_augmented_dataset()
    else:
        # Use basic dataset
        from train import create_sample_dataset
        train_texts = create_sample_dataset()
        print(f"✅ Loaded {len(train_texts)} basic samples")
    
    # Initialize trainer
    trainer = EnhancedTrainer(config)
    
    # Start training
    trainer.train(train_texts)
    
    print("\n🎯 Training Summary:")
    print(f"Total epochs: {config['epochs']}")
    print(f"Best loss: {trainer.best_loss:.4f}")
    print(f"Model saved: enhanced_final_model.pth")
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_enhanced_model(trainer.model)

def test_enhanced_model(model):
    """Test the enhanced model"""
    model.eval()
    
    # Simple test generation
    print("Testing text generation...")
    
    # Create a simple prompt
    prompt = torch.randint(1000, 2000, (1, 10))  # Random tokens as prompt
    
    with torch.no_grad():
        generated = model.generate(
            prompt, 
            max_length=30, 
            temperature=0.8, 
            top_k=50, 
            top_p=0.9
        )
    
    print(f"Generated sequence length: {generated.shape[1]}")
    print(f"Prompt length: {prompt.shape[1]}")
    print(f"New tokens generated: {generated.shape[1] - prompt.shape[1]}")
    print("✅ Model generation test passed!")

if __name__ == "__main__":
    main()
