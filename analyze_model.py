import torch
import json
import os
from model import SmallLLM
import pickle

def analyze_model():
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print("=== VAANI LLM SPECIFICATIONS ===\n")
    
    # Model Architecture
    print("MODEL ARCHITECTURE:")
    print(f"- Model Name: Vaani LLM")
    print(f"- Architecture: Transformer-based Language Model")
    print(f"- Vocabulary Size: {config['vocab_size']:,}")
    print(f"- Hidden Size (d_model): {config['hidden_size']}")
    print(f"- Number of Layers: {config['num_layers']}")
    print(f"- Number of Attention Heads: {config['num_heads']}")
    print(f"- Feed Forward Dimension: {config['intermediate_size']:,}")
    print(f"- Maximum Sequence Length: {config['max_seq_len']}")
    print(f"- Dropout Rate: {config['dropout']}")
    
    # Create model to count parameters
    model = SmallLLM(
        vocab_size=config['vocab_size'],
        d_model=config['hidden_size'],
        n_layers=config['num_layers'],
        n_heads=config['num_heads'],
        d_ff=config['intermediate_size'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    total_params = model.count_parameters()
    
    print(f"\nMODEL PARAMETERS:")
    print(f"- Total Parameters: {total_params:,}")
    print(f"- Trainable Parameters: {total_params:,}")
    print(f"- Model Size: ~{total_params/1e6:.1f}M parameters")
    
    # Calculate memory requirements
    param_memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"- Memory for Parameters (FP32): ~{param_memory_mb:.1f} MB")
    print(f"- Memory for Parameters (FP16): ~{param_memory_mb/2:.1f} MB")
    
    # Check if model file exists and get its size
    if os.path.exists('model.pth'):
        model_size_mb = os.path.getsize('model.pth') / (1024 * 1024)
        print(f"- Saved Model File Size: {model_size_mb:.1f} MB")
    
    # Training Configuration
    print(f"\nTRAINING CONFIGURATION:")
    print(f"- Batch Size: {config['batch_size']}")
    print(f"- Learning Rate: {config['learning_rate']}")
    print(f"- Gradient Accumulation Steps: {config['gradient_accumulation_steps']}")
    
    # Tokenizer info
    if os.path.exists('tokenizer.pkl'):
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer_data = pickle.load(f)
            actual_vocab_size = len(tokenizer_data['word_to_id'])
            print(f"\nTOKENIZER SPECIFICATIONS:")
            print(f"- Type: Simple Word-based Tokenizer")
            print(f"- Actual Vocabulary Size: {actual_vocab_size:,}")
            print(f"- Special Tokens: {list(tokenizer_data['special_tokens'].keys())}")
    
    # Model Features
    print(f"\nMODEL FEATURES:")
    print(f"- Rotary Position Encoding (RoPE): Yes")
    print(f"- Pre-layer Normalization: Yes")
    print(f"- Weight Tying: Yes (embedding and output layer)")
    print(f"- Activation Function: GELU")
    print(f"- Attention Type: Multi-head Self-attention with Causal Masking")
    
    # Performance estimates
    print(f"\nPERFORMANCE ESTIMATES:")
    seq_len = config['max_seq_len']
    batch_size = config['batch_size']
    
    # Rough FLOP calculation for inference
    # Attention: 4 * batch_size * seq_len^2 * hidden_size * num_layers
    # FFN: 8 * batch_size * seq_len * hidden_size * ff_size * num_layers
    attention_flops = 4 * batch_size * seq_len * seq_len * config['hidden_size'] * config['num_layers']
    ffn_flops = 8 * batch_size * seq_len * config['hidden_size'] * config['intermediate_size'] * config['num_layers']
    total_flops = attention_flops + ffn_flops
    
    print(f"- Approximate FLOPs per forward pass: {total_flops/1e9:.2f} GFLOPs")
    print(f"- Context Window: {config['max_seq_len']} tokens")
    
    return model, config

if __name__ == "__main__":
    analyze_model()
