# 🔧 Vaani LLM Configuration Guide

This document explains all configuration files and parameters used in the Vaani LLM project.

## 📋 Configuration Files Overview

### Primary Configuration Files
- **`config.json`** - Main model architecture configuration
- **`requirements.txt`** - Python dependencies
- **`MODEL_SPECIFICATIONS.md`** - Detailed technical specifications

## 📄 config.json - Model Configuration

### Current Configuration
```json
{
  "vocab_size": 16000,
  "hidden_size": 1408,
  "num_layers": 16,
  "num_heads": 11,
  "intermediate_size": 5632,
  "max_seq_len": 1024,
  "batch_size": 1,
  "learning_rate": 0.0002,
  "dropout": 0.1,
  "gradient_accumulation_steps": 4
}
```

### Parameter Explanations

#### Model Architecture Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 16000 | Maximum vocabulary size (actual: 262) |
| `hidden_size` | 1408 | Model hidden dimension (d_model) |
| `num_layers` | 16 | Number of transformer layers |
| `num_heads` | 11 | Number of attention heads per layer |
| `intermediate_size` | 5632 | Feed-forward network hidden size |
| `max_seq_len` | 1024 | Maximum sequence length |
| `dropout` | 0.1 | Dropout probability (10%) |

#### Training Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 1 | Base batch size |
| `learning_rate` | 0.0002 | Adam optimizer learning rate |
| `gradient_accumulation_steps` | 4 | Gradient accumulation for effective batch size |

### Effective Configuration
```json
{
  "effective_batch_size": 4,
  "parameter_count": 427442560,
  "model_size_gb": 1.63,
  "context_window": 1024,
  "attention_head_dim": 128
}
```

## 🎯 Configuration Recommendations

### For Improved Performance

#### Larger Training Setup
```json
{
  "vocab_size": 32000,
  "hidden_size": 1408,
  "num_layers": 16,
  "num_heads": 16,
  "intermediate_size": 5632,
  "max_seq_len": 2048,
  "batch_size": 4,
  "learning_rate": 0.0001,
  "dropout": 0.1,
  "gradient_accumulation_steps": 8
}
```

#### Production Setup
```json
{
  "vocab_size": 50000,
  "hidden_size": 2048,
  "num_layers": 24,
  "num_heads": 16,
  "intermediate_size": 8192,
  "max_seq_len": 2048,
  "batch_size": 2,
  "learning_rate": 0.00005,
  "dropout": 0.1,
  "gradient_accumulation_steps": 16
}
```

## 🔄 Configuration Validation

### Architecture Constraints
```python
def validate_config(config):
    # Ensure head dimension is reasonable
    head_dim = config['hidden_size'] // config['num_heads']
    assert head_dim >= 32, f"Head dimension too small: {head_dim}"
    assert head_dim <= 256, f"Head dimension too large: {head_dim}"
    
    # Ensure divisibility
    assert config['hidden_size'] % config['num_heads'] == 0, \
           "hidden_size must be divisible by num_heads"
    
    # Check FFN ratio
    ffn_ratio = config['intermediate_size'] / config['hidden_size']
    assert 2 <= ffn_ratio <= 8, f"FFN ratio should be 2-8x: {ffn_ratio}"
    
    print("✅ Configuration validated successfully")
```

### Memory Estimation
```python
def estimate_memory(config):
    params = estimate_parameters(config)
    
    model_memory_gb = (params * 4) / (1024**3)  # FP32
    gradient_memory_gb = model_memory_gb  # Same as model
    optimizer_memory_gb = model_memory_gb * 2  # Adam states
    
    total_training_gb = model_memory_gb + gradient_memory_gb + optimizer_memory_gb
    
    print(f"Model: {model_memory_gb:.2f} GB")
    print(f"Training: {total_training_gb:.2f} GB")
    
    return total_training_gb

def estimate_parameters(config):
    embedding = config['vocab_size'] * config['hidden_size']
    
    # Per layer
    attention = config['hidden_size'] * config['hidden_size'] * 4  # Q, K, V, O
    ffn = config['hidden_size'] * config['intermediate_size'] * 2  # Up, Down
    layer_norm = config['hidden_size'] * 4  # 2 norms per layer, weight + bias
    
    per_layer = attention + ffn + layer_norm
    total_layers = per_layer * config['num_layers']
    
    final_norm = config['hidden_size'] * 2
    
    # Output head is tied with embedding, so no extra params
    total = embedding + total_layers + final_norm
    
    return total
```

## 🛠️ Environment Configuration

### Python Dependencies (requirements.txt)
```txt
# Core Framework
torch>=2.0.0
numpy>=1.21.0

# Training Utilities
tqdm>=4.62.0
datasets>=2.0.0

# Optional Enhancements
transformers>=4.20.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
scikit-learn>=1.0.0
psutil>=5.8.0
accelerate>=0.20.0

# Development
pytest>=6.2.0
jupyter>=1.0.0
ipython>=8.0.0
```

### System Requirements
```yaml
minimum_requirements:
  python: ">=3.8"
  ram: "8 GB"
  storage: "5 GB"
  gpu_memory: "4 GB (optional)"

recommended_requirements:
  python: ">=3.9"
  ram: "16 GB"
  storage: "10 GB"
  gpu_memory: "8 GB"

optimal_requirements:
  python: ">=3.10"
  ram: "32 GB"
  storage: "20 GB"
  gpu_memory: "16 GB+"
```

## 🎛️ Runtime Configuration

### Inference Settings
```python
INFERENCE_CONFIG = {
    "device": "auto",  # "cpu", "cuda", or "auto"
    "batch_size": 1,
    "max_new_tokens": 50,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": 0,
    "eos_token_id": 3,
    "use_cache": True
}
```

### Training Settings
```python
TRAINING_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mixed_precision": "fp16",
    "dataloader_num_workers": 4,
    "pin_memory": True,
    "gradient_checkpointing": False,
    "save_steps": 100,
    "eval_steps": 50,
    "logging_steps": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0
}
```

## 📊 Performance Tuning

### Memory Optimization
```python
# For low memory environments
LOW_MEMORY_CONFIG = {
    "gradient_checkpointing": True,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "dataloader_num_workers": 1,
    "pin_memory": False,
    "mixed_precision": "fp16"
}

# For high memory environments  
HIGH_MEMORY_CONFIG = {
    "gradient_checkpointing": False,
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "dataloader_num_workers": 8,
    "pin_memory": True,
    "mixed_precision": "bf16"
}
```

### Speed Optimization
```python
# For fastest training
SPEED_CONFIG = {
    "compile_model": True,  # PyTorch 2.0 compile
    "use_flash_attention": True,  # If available
    "dataloader_num_workers": 8,
    "pin_memory": True,
    "non_blocking": True,
    "channels_last": False
}
```

## 🔍 Configuration Testing

### Validate Before Training
```python
def test_configuration():
    import torch
    from model import SmallLLM
    import json
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Test model creation
    model = SmallLLM(**{k: v for k, v in config.items() 
                       if k in ['vocab_size', 'hidden_size', 'num_layers', 
                               'num_heads', 'intermediate_size', 'max_seq_len', 'dropout']})
    
    # Test forward pass
    dummy_input = torch.randint(0, config['vocab_size'], (1, 10))
    
    try:
        output = model(dummy_input)
        print(f"✅ Model forward pass successful")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Total parameters: {model.count_parameters():,}")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_configuration()
```

## 🚀 Quick Configuration Updates

### To Increase Model Size
```bash
# Update config.json to scale up
python -c "
import json
with open('config.json', 'r') as f: config = json.load(f)
config['hidden_size'] = 2048
config['num_layers'] = 24
config['intermediate_size'] = 8192
with open('config.json', 'w') as f: json.dump(config, f, indent=2)
print('✅ Configuration updated for larger model')
"
```

### To Optimize for Memory
```bash
# Update for low memory training
python -c "
import json
with open('config.json', 'r') as f: config = json.load(f)
config['batch_size'] = 1
config['gradient_accumulation_steps'] = 8
config['max_seq_len'] = 512
with open('config.json', 'w') as f: json.dump(config, f, indent=2)
print('✅ Configuration optimized for low memory')
"
```

---

This configuration guide provides comprehensive documentation for customizing and optimizing your Vaani LLM setup across different hardware and use case scenarios.
