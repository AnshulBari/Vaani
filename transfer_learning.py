"""
Transfer Learning for Vaani LLM
Initialize your model with pre-trained weights from other models
"""

import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, AutoModel, AutoTokenizer
from model import SmallLLM
import numpy as np

class TransferLearningInitializer:
    def __init__(self, source_model_name="gpt2"):
        """
        Initialize transfer learning from a pre-trained model
        
        Args:
            source_model_name: HuggingFace model name to transfer from
        """
        self.source_model_name = source_model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading source model: {source_model_name}")
        self.source_model = GPT2Model.from_pretrained(source_model_name)
        self.source_tokenizer = GPT2Tokenizer.from_pretrained(source_model_name)
        
        print(f"Source model loaded with {sum(p.numel() for p in self.source_model.parameters()):,} parameters")
    
    def transfer_embeddings(self, target_model, target_tokenizer):
        """Transfer embedding weights from source to target model"""
        print("Transferring embedding weights...")
        
        source_embeddings = self.source_model.wte.weight.data  # Word embeddings
        target_embeddings = target_model.token_embedding.weight.data
        
        # Get common vocabulary
        transferred_count = 0
        
        for word, target_idx in target_tokenizer['word_to_id'].items():
            # Try to find word in source tokenizer
            source_tokens = self.source_tokenizer.encode(word, add_special_tokens=False)
            
            if len(source_tokens) == 1:  # Single token match
                source_idx = source_tokens[0]
                if source_idx < source_embeddings.size(0):
                    # Copy embedding
                    target_embeddings[target_idx] = source_embeddings[source_idx]
                    transferred_count += 1
        
        print(f"Transferred {transferred_count}/{len(target_tokenizer['word_to_id'])} embeddings")
        return transferred_count
    
    def transfer_attention_weights(self, target_model):
        """Transfer attention weights with dimension adaptation"""
        print("Transferring attention weights...")
        
        source_layers = self.source_model.h  # Transformer blocks
        target_layers = target_model.layers
        
        num_layers_to_transfer = min(len(source_layers), len(target_layers))
        
        for i in range(num_layers_to_transfer):
            source_layer = source_layers[i]
            target_layer = target_layers[i]
            
            # Transfer attention weights
            source_attn = source_layer.attn
            target_attn = target_layer.attention
            
            # Adapt query, key, value projections
            self._adapt_linear_layer(source_attn.c_attn, target_attn.q_proj, "q")
            self._adapt_linear_layer(source_attn.c_attn, target_attn.k_proj, "k") 
            self._adapt_linear_layer(source_attn.c_attn, target_attn.v_proj, "v")
            self._adapt_linear_layer(source_attn.c_proj, target_attn.o_proj, "o")
            
            # Transfer feed-forward weights
            source_mlp = source_layer.mlp
            target_ff = target_layer.feed_forward
            
            self._adapt_linear_layer(source_mlp.c_fc, target_ff.linear1, "ff1")
            self._adapt_linear_layer(source_mlp.c_proj, target_ff.linear2, "ff2")
            
            # Transfer layer norms
            target_layer.norm1.weight.data = source_layer.ln_1.weight.data.clone()
            target_layer.norm1.bias.data = source_layer.ln_1.bias.data.clone()
            target_layer.norm2.weight.data = source_layer.ln_2.weight.data.clone()
            target_layer.norm2.bias.data = source_layer.ln_2.bias.data.clone()
        
        print(f"Transferred weights for {num_layers_to_transfer} layers")
    
    def _adapt_linear_layer(self, source_layer, target_layer, layer_type):
        """Adapt linear layer weights between different dimensions"""
        source_weight = source_layer.weight.data
        source_bias = source_layer.bias.data if source_layer.bias is not None else None
        
        target_weight = target_layer.weight.data
        target_bias = target_layer.bias.data if target_layer.bias is not None else None
        
        # For GPT-2, c_attn contains Q, K, V concatenated
        if layer_type in ["q", "k", "v"] and "c_attn" in str(source_layer):
            # Split the concatenated QKV weights
            split_size = source_weight.size(0) // 3
            if layer_type == "q":
                source_weight = source_weight[:split_size]
                source_bias = source_bias[:split_size] if source_bias is not None else None
            elif layer_type == "k":
                source_weight = source_weight[split_size:2*split_size]
                source_bias = source_bias[split_size:2*split_size] if source_bias is not None else None
            elif layer_type == "v":
                source_weight = source_weight[2*split_size:]
                source_bias = source_bias[2*split_size:] if source_bias is not None else None
        
        # Adapt dimensions
        if source_weight.shape == target_weight.shape:
            # Direct copy
            target_weight.copy_(source_weight)
            if target_bias is not None and source_bias is not None:
                target_bias.copy_(source_bias)
        else:
            # Dimension adaptation
            min_out = min(source_weight.size(0), target_weight.size(0))
            min_in = min(source_weight.size(1), target_weight.size(1))
            
            target_weight[:min_out, :min_in] = source_weight[:min_out, :min_in]
            
            if target_bias is not None and source_bias is not None:
                target_bias[:min_out] = source_bias[:min_out]
    
    def initialize_target_model(self, target_model, target_tokenizer):
        """Complete transfer learning initialization"""
        print("Starting transfer learning initialization...")
        
        # Transfer embeddings
        self.transfer_embeddings(target_model, target_tokenizer)
        
        # Transfer attention and feed-forward weights
        self.transfer_attention_weights(target_model)
        
        # Initialize final layer norm
        if hasattr(self.source_model, 'ln_f'):
            target_model.norm.weight.data = self.source_model.ln_f.weight.data.clone()
            target_model.norm.bias.data = self.source_model.ln_f.bias.data.clone()
        
        print("Transfer learning initialization completed!")

def initialize_with_pretrained():
    """Initialize Vaani model with pre-trained weights"""
    
    # Load your tokenizer
    import pickle
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    print(f"Target vocabulary size: {len(tokenizer['word_to_id'])}")
    
    # Create target model
    model = SmallLLM(
        vocab_size=len(tokenizer['word_to_id']),
        d_model=1024,
        n_layers=16,
        n_heads=16,
        d_ff=4096,
        max_seq_len=512,
        dropout=0.1
    )
    
    print(f"Target model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize with transfer learning
    initializer = TransferLearningInitializer("gpt2")  # or "gpt2-medium"
    initializer.initialize_target_model(model, tokenizer)
    
    # Save initialized model
    torch.save(model.state_dict(), 'pretrained_initialized_model.pth')
    print("Initialized model saved as: pretrained_initialized_model.pth")
    
    return model, tokenizer

def fine_tune_on_custom_data():
    """Fine-tune the initialized model on your custom data"""
    
    # Load initialized model
    model, tokenizer = initialize_with_pretrained()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Custom fine-tuning data (expand this!)
    fine_tuning_data = [
        "Vaani is a small language model designed for efficient text generation.",
        "The model uses transformer architecture with multi-head attention.",
        "Knowledge distillation can improve small model performance significantly.",
        "Transfer learning helps bootstrap training with pre-trained weights.",
        "Fine-tuning adapts general models to specific domains and tasks.",
        "Small models are more efficient for deployment in resource-constrained environments.",
        "The attention mechanism allows models to focus on relevant context.",
        "Transformer blocks stack multiple layers of self-attention and feed-forward networks.",
        "Language models learn to predict the next word given previous context.",
        "Pre-training on large corpora provides general language understanding."
    ]
    
    # Fine-tuning configuration
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    print("Starting fine-tuning...")
    
    model.train()
    for epoch in range(3):
        total_loss = 0
        
        for text in fine_tuning_data:
            optimizer.zero_grad()
            
            # Tokenize
            tokens = []
            for word in text.lower().split():
                tokens.append(tokenizer['word_to_id'].get(word, tokenizer['word_to_id']['<unk>']))
            
            if len(tokens) > 1:
                input_ids = torch.tensor(tokens[:-1]).unsqueeze(0).to(device)
                targets = torch.tensor(tokens[1:]).unsqueeze(0).to(device)
                
                # Forward pass
                outputs = model(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[1]
                else:
                    logits = outputs
                
                # Compute loss
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1)
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(fine_tuning_data)
        print(f"Epoch {epoch+1}: Average loss = {avg_loss:.4f}")
    
    # Save fine-tuned model
    torch.save(model.state_dict(), 'fine_tuned_model.pth')
    print("Fine-tuned model saved as: fine_tuned_model.pth")

if __name__ == "__main__":
    # Option 1: Just initialize with pre-trained weights
    initialize_with_pretrained()
    
    # Option 2: Initialize and fine-tune
    # fine_tune_on_custom_data()
