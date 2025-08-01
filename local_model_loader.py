#!/usr/bin/env python3
"""
Local Model Loader for Vaani 1.5B Parameter Model
Load and use your trained Vaani model locally with the downloaded files.
"""

import torch
import torch.nn as nn
import json
import os
import numpy as np
from typing import Dict, List, Optional

class VaaniLocalModel:
    """Local implementation of Vaani model for inference"""
    
    def __init__(self, model_path: str, config_path: str):
        """
        Initialize the local Vaani model
        
        Args:
            model_path: Path to the model.pth file
            config_path: Path to the config.json file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        print(f"📋 Model config loaded: {self.config['hidden_size']}D, {self.config['num_layers']} layers")
        
        # Create model architecture
        self.model = self._create_model()
        
        # Load trained weights
        self._load_weights(model_path)
        
        # Set up tokenizer
        self._setup_tokenizer()
        
        print(f"✅ Vaani 1.5B model loaded successfully!")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load model configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    
    def _create_model(self):
        """Create the transformer model architecture"""
        
        class SimpleTransformerLM(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.vocab_size = config['vocab_size']
                self.hidden_size = config['hidden_size']
                
                self.embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
                self.pos_embedding = nn.Embedding(config['max_seq_len'], config['hidden_size'])
                
                from torch.nn import TransformerEncoder, TransformerEncoderLayer
                encoder_layers = TransformerEncoderLayer(
                    d_model=config['hidden_size'],
                    nhead=config['num_heads'],
                    dim_feedforward=config['intermediate_size'],
                    dropout=config['dropout'],
                    batch_first=True
                )
                
                self.transformer = TransformerEncoder(encoder_layers, config['num_layers'])
                self.ln_f = nn.LayerNorm(config['hidden_size'])
                self.head = nn.Linear(config['hidden_size'], config['vocab_size'])
                
            def forward(self, input_ids):
                seq_len = input_ids.size(1)
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                
                x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
                x = self.transformer(x)
                x = self.ln_f(x)
                logits = self.head(x)
                
                return logits
        
        model = SimpleTransformerLM(self.config).to(self.device)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📊 Model parameters: {param_count:,} (~{param_count/1_000_000:.1f}M)")
        
        return model
    
    def _load_weights(self, model_path: str):
        """Load the trained model weights"""
        print(f"📂 Loading weights from: {model_path}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✅ Weights loaded successfully!")
    
    def _setup_tokenizer(self):
        """Set up the simple tokenizer (same as training)"""
        self.vocab = {i: f"token_{i}" for i in range(self.config['vocab_size'])}
        self.vocab.update({
            0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>',
            4: 'the', 5: 'a', 6: 'and', 7: 'to', 8: 'of', 9: 'in',
            10: 'is', 11: 'that', 12: 'for', 13: 'with', 14: 'as',
            15: 'it', 16: 'on', 17: 'be', 18: 'at', 19: 'by'
        })
        
        # Create reverse mapping
        self.token_to_id = {v: k for k, v in self.vocab.items() if not v.startswith('token_')}
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        words = text.lower().split()
        tokens = [2]  # Start with <bos>
        
        for word in words[:10]:  # Limit to 10 words
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                # Use a consistent hash for unknown words
                token_id = (hash(word) % (self.config['vocab_size'] - 100)) + 100
                tokens.append(token_id)
        
        return tokens
    
    def detokenize(self, tokens: List[int]) -> str:
        """Convert token IDs back to text"""
        words = []
        for token in tokens:
            if token in self.vocab:
                word = self.vocab[token]
                if not word.startswith('<') and not word.startswith('token_'):
                    words.append(word)
        return ' '.join(words)
    
    def generate(self, prompt: str, max_length: int = 20, temperature: float = 0.8, 
                 top_k: Optional[int] = None, top_p: Optional[float] = None) -> str:
        """
        Generate text continuation for the given prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top k tokens for sampling
            top_p: Keep only tokens with cumulative probability <= top_p
            
        Returns:
            Generated text continuation
        """
        self.model.eval()
        
        # Tokenize prompt
        input_ids = self.tokenize(prompt)
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input (last max_seq_len tokens)
                if len(generated) > self.config['max_seq_len']:
                    current_input = torch.tensor([generated[-self.config['max_seq_len']:]], device=self.device)
                else:
                    current_input = torch.tensor([generated], device=self.device)
                
                # Get predictions
                logits = self.model(current_input)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                generated.append(next_token)
                
                # Stop if EOS token
                if next_token == 3:  # <eos>
                    break
        
        # Decode only the generated part
        generated_part = generated[len(input_ids):]
        return self.detokenize(generated_part)
    
    def analyze_prompt(self, prompt: str) -> Dict:
        """Analyze how the model processes a prompt"""
        self.model.eval()
        
        tokens = self.tokenize(prompt)
        input_tensor = torch.tensor([tokens], device=self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            last_token_logits = logits[0, -1, :]
            
            # Get top predictions
            top_probs, top_indices = torch.topk(torch.softmax(last_token_logits, dim=-1), 10)
            
            return {
                'input_tokens': tokens,
                'token_meanings': [self.vocab.get(t, f'unknown_{t}') for t in tokens],
                'top_predictions': [
                    {
                        'token_id': idx.item(),
                        'token': self.vocab.get(idx.item(), f'token_{idx.item()}'),
                        'probability': prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]
            }


def demo_local_model(model_dir: str):
    """Demo function to test the local model"""
    
    model_path = os.path.join(model_dir, 'model.pth')
    config_path = os.path.join(model_dir, 'config.json')
    
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"❌ Model files not found in {model_dir}")
        print(f"   Looking for: model.pth and config.json")
        return
    
    print("🚀 Loading Vaani 1.5B Local Model")
    print("=" * 50)
    
    # Load model
    model = VaaniLocalModel(model_path, config_path)
    
    # Test prompts
    test_prompts = [
        "artificial intelligence will",
        "the future of science",
        "technology enables us",
        "learning new skills"
    ]
    
    print(f"\n🎯 Testing Text Generation:")
    print("-" * 30)
    
    for prompt in test_prompts:
        print(f"\n👤 Prompt: '{prompt}'")
        
        # Try different sampling strategies
        results = []
        
        try:
            # Greedy (low temperature)
            result1 = model.generate(prompt, max_length=15, temperature=0.1)
            results.append(f"Greedy: '{result1}'")
            
            # Balanced
            result2 = model.generate(prompt, max_length=15, temperature=0.8)
            results.append(f"Balanced: '{result2}'")
            
            # Creative (high temperature)
            result3 = model.generate(prompt, max_length=15, temperature=1.2)
            results.append(f"Creative: '{result3}'")
            
            # Top-k sampling
            result4 = model.generate(prompt, max_length=15, temperature=0.8, top_k=50)
            results.append(f"Top-k: '{result4}'")
            
        except Exception as e:
            results.append(f"Error: {e}")
        
        for result in results:
            print(f"   🤖 {result}")
    
    # Analyze a prompt
    print(f"\n🔍 Analyzing Prompt: 'artificial intelligence'")
    analysis = model.analyze_prompt("artificial intelligence")
    
    print(f"Input tokens: {analysis['input_tokens']}")
    print(f"Token meanings: {analysis['token_meanings']}")
    print(f"Top predictions:")
    for i, pred in enumerate(analysis['top_predictions'][:5]):
        print(f"   {i+1}. {pred['token']} (ID: {pred['token_id']}) - {pred['probability']:.4f}")
    
    print(f"\n✅ Local model testing completed!")
    
    return model


if __name__ == "__main__":
    # Example usage
    model_directory = "."  # Current directory
    demo_local_model(model_directory)
