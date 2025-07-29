"""
Knowledge Distillation for Vaani LLM
Train your small model using a larger teacher model (like GPT-2, Llama, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model import SmallLLM
from tokenizer import SimpleTokenizer
import numpy as np
from tqdm import tqdm

class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model_name="gpt2", temperature=4.0, alpha=0.7):
        """
        Initialize Knowledge Distillation trainer
        
        Args:
            teacher_model_name: HuggingFace model name (gpt2, gpt2-medium, etc.)
            temperature: Temperature for softmax (higher = softer targets)
            alpha: Weight for distillation loss vs ground truth loss
        """
        self.temperature = temperature
        self.alpha = alpha
        
        # Load teacher model (GPT-2)
        print(f"Loading teacher model: {teacher_model_name}")
        self.teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_name)
        self.teacher_tokenizer = GPT2Tokenizer.from_pretrained(teacher_model_name)
        self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.teacher_model.to(device)
        self.device = device
        
        print(f"Teacher model loaded on {device}")
        print(f"Teacher parameters: {sum(p.numel() for p in self.teacher_model.parameters()):,}")
    
    def get_teacher_predictions(self, text_batch):
        """Get soft targets from teacher model"""
        # Tokenize for teacher
        teacher_inputs = self.teacher_tokenizer(
            text_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
            # Apply temperature to get soft targets
            teacher_logits = teacher_outputs.logits / self.temperature
            teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        return teacher_probs, teacher_inputs.input_ids
    
    def distillation_loss(self, student_logits, teacher_probs, true_labels):
        """
        Compute combined distillation loss
        
        Args:
            student_logits: Raw logits from student model
            teacher_probs: Soft targets from teacher model  
            true_labels: Ground truth tokens
        """
        # Soft target loss (KL divergence)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        kl_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        
        # Hard target loss (standard cross-entropy)
        ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), 
                                  true_labels.view(-1), ignore_index=-100)
        
        # Combined loss
        total_loss = self.alpha * (self.temperature ** 2) * kl_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, kl_loss, ce_loss
    
    def train_step(self, student_model, student_tokenizer, text_batch, optimizer):
        """Single training step with knowledge distillation"""
        
        # Get teacher predictions
        teacher_probs, teacher_tokens = self.get_teacher_predictions(text_batch)
        
        # Convert teacher tokens to student vocabulary
        student_texts = []
        for i in range(teacher_tokens.size(0)):
            # Decode teacher tokens back to text
            text = self.teacher_tokenizer.decode(teacher_tokens[i], skip_special_tokens=True)
            student_texts.append(text)
        
        # Tokenize for student model
        student_tokens = []
        for text in student_texts:
            tokens = student_tokenizer.encode(text)[:512]  # Truncate to max length
            # Pad to match teacher length
            if len(tokens) < teacher_tokens.size(1):
                tokens.extend([student_tokenizer.special_tokens['<pad>']] * 
                             (teacher_tokens.size(1) - len(tokens)))
            student_tokens.append(tokens[:teacher_tokens.size(1)])
        
        student_input_ids = torch.tensor(student_tokens).to(self.device)
        
        # Forward pass through student
        student_model.train()
        optimizer.zero_grad()
        
        student_outputs = student_model(student_input_ids)
        if isinstance(student_outputs, tuple):
            student_logits = student_outputs[1]  # Skip loss, get logits
        else:
            student_logits = student_outputs
        
        # Align dimensions
        min_seq_len = min(student_logits.size(1), teacher_probs.size(1))
        student_logits = student_logits[:, :min_seq_len, :]
        teacher_probs_aligned = teacher_probs[:, :min_seq_len, :]
        
        # Map teacher vocabulary to student vocabulary
        teacher_probs_mapped = self.map_teacher_to_student_vocab(
            teacher_probs_aligned, student_tokenizer
        )
        
        # Compute distillation loss
        total_loss, kl_loss, ce_loss = self.distillation_loss(
            student_logits, teacher_probs_mapped, student_input_ids[:, :min_seq_len]
        )
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        optimizer.step()
        
        return total_loss.item(), kl_loss.item(), ce_loss.item()
    
    def map_teacher_to_student_vocab(self, teacher_probs, student_tokenizer):
        """Map teacher vocabulary probabilities to student vocabulary"""
        batch_size, seq_len, teacher_vocab_size = teacher_probs.shape
        student_vocab_size = len(student_tokenizer.word_to_id)
        
        # Create mapping from teacher tokens to student tokens
        mapped_probs = torch.zeros(batch_size, seq_len, student_vocab_size).to(self.device)
        
        # Simple mapping: use most common words
        for teacher_idx in range(min(1000, teacher_vocab_size)):  # Map top 1000 tokens
            teacher_token = self.teacher_tokenizer.convert_ids_to_tokens([teacher_idx])[0]
            
            # Clean token (remove special characters)
            clean_token = teacher_token.replace('Ġ', '').lower()
            
            if clean_token in student_tokenizer.word_to_id:
                student_idx = student_tokenizer.word_to_id[clean_token]
                mapped_probs[:, :, student_idx] += teacher_probs[:, :, teacher_idx]
            else:
                # Map to unknown token
                unk_idx = student_tokenizer.word_to_id.get('<unk>', 1)
                mapped_probs[:, :, unk_idx] += teacher_probs[:, :, teacher_idx]
        
        # Renormalize
        mapped_probs = F.softmax(mapped_probs, dim=-1)
        
        return mapped_probs

def distill_model():
    """Main distillation training function"""
    
    # Initialize distillation trainer
    distiller = KnowledgeDistillationTrainer(
        teacher_model_name="gpt2",  # You can use gpt2-medium for better teacher
        temperature=4.0,
        alpha=0.7
    )
    
    # Load your student model
    print("Loading student model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    import pickle
    with open('tokenizer.pkl', 'rb') as f:
        student_tokenizer = pickle.load(f)
    
    # Create student model
    student_model = SmallLLM(
        vocab_size=len(student_tokenizer['word_to_id']),
        d_model=1024,
        n_layers=16,
        n_heads=16,
        d_ff=4096,
        max_seq_len=512,
        dropout=0.1
    ).to(device)
    
    # Load pre-trained weights if available
    try:
        student_model.load_state_dict(torch.load('final_model.pth', map_location=device))
        print("Loaded pre-trained student weights")
    except:
        print("Starting from random weights")
    
    # Optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Training data (you can expand this)
    training_texts = [
        "The future of artificial intelligence looks very promising.",
        "Machine learning models are becoming more sophisticated every year.",
        "Deep learning has revolutionized computer vision and natural language processing.",
        "Neural networks can learn complex patterns from large datasets.",
        "Transformers have become the dominant architecture for language models.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Large language models can generate human-like text.",
        "Knowledge distillation helps transfer information from large to small models.",
        "Fine-tuning allows models to adapt to specific tasks and domains.",
        "The democratization of AI will benefit society in many ways."
    ]
    
    print(f"Starting knowledge distillation training...")
    print(f"Student parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    # Training loop
    num_epochs = 5
    batch_size = 2
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_kl_loss = 0
        total_ce_loss = 0
        num_batches = 0
        
        # Create batches
        for i in range(0, len(training_texts), batch_size):
            batch = training_texts[i:i+batch_size]
            
            loss, kl_loss, ce_loss = distiller.train_step(
                student_model, student_tokenizer, batch, optimizer
            )
            
            total_loss += loss
            total_kl_loss += kl_loss
            total_ce_loss += ce_loss
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}: "
                      f"Total: {loss:.4f}, KL: {kl_loss:.4f}, CE: {ce_loss:.4f}")
        
        avg_loss = total_loss / num_batches
        avg_kl = total_kl_loss / num_batches
        avg_ce = total_ce_loss / num_batches
        
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, KL: {avg_kl:.4f}, CE: {avg_ce:.4f}")
        
        # Save checkpoint
        torch.save(student_model.state_dict(), f'distilled_model_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(student_model.state_dict(), 'distilled_final_model.pth')
    print("Knowledge distillation completed!")
    print("Model saved as: distilled_final_model.pth")

if __name__ == "__main__":
    distill_model()
