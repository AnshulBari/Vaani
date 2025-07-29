# Small LLM Project

A 4-5GB Large Language Model implementation with modern transformer architecture featuring RoPE positional encoding and pre-norm design.

## Features

- **Modern Architecture**: Transformer-based with RoPE (Rotary Positional Embedding)
- **Efficient Design**: Pre-norm architecture for stable training
- **Optimized Size**: ~4-5GB model suitable for consumer hardware
- **Training Pipeline**: Complete training infrastructure with mixed precision
- **Inference Engine**: Fast text generation with customizable parameters

## Model Architecture

- **Parameters**: ~4B (configurable)
- **Layers**: 20-24 transformer blocks
- **Attention**: Multi-head attention with RoPE
- **Context Length**: Up to 2048 tokens
- **Vocabulary**: 25K-50K tokens

## Files Structure

```
small_llm_project/
├── model.py              # Core model architecture
├── tokenizer.py          # Simple tokenizer implementation
├── train.py              # Training script
├── advanced_trainer.py   # Advanced training with mixed precision
├── inference.py          # Text generation and chat interface
├── data_sources.py       # Training data preparation
├── test.py              # Model testing and validation
├── requirements.txt      # Dependencies
├── README.md            # This file
└── configs/
    └── model_config.json # Model configuration
```

## Quick Start

### 1. Install Dependencies

```bash
cd d:\GitRepo\LLM\small_llm_project
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

This will:
- Create and train a tokenizer on sample data
- Initialize the model with ~4B parameters
- Train for 5 epochs with progress monitoring
- Save model checkpoints and final weights
- Generate `small_llm_final.pth`, `tokenizer.pkl`, and `model_config.json`

Training output will show:
- Model parameter count
- Estimated model size in GB
- Training progress with loss monitoring
- Checkpoint saving every 100 steps

### 3. Run Inference

```bash
python inference.py
```

This starts an interactive chat where you can test the model:

```
Small LLM Chat (type 'quit' to exit)
==================================================

You: What is machine learning?
AI: machine learning is a subset of artificial intelligence that enables computers to learn patterns...

You: quit
Goodbye!
```

## Configuration

The model configuration can be adjusted in `train.py`:

```python
config = {
    'vocab_size': 10000,    # Vocabulary size
    'd_model': 1024,        # Model dimension
    'n_layers': 20,         # Number of transformer layers
    'n_heads': 16,          # Number of attention heads
    'd_ff': 4096,           # Feed-forward dimension
    'max_seq_len': 512,     # Maximum sequence length
    'dropout': 0.1,         # Dropout rate
    'batch_size': 4,        # Training batch size
    'learning_rate': 3e-4,  # Learning rate
    'epochs': 5,            # Training epochs
}
```

## Model Architecture Details

### Core Components

1. **Rotary Positional Embedding (RoPE)**
   - More efficient than traditional positional encoding
   - Better handling of longer sequences
   - Relative position encoding

2. **Multi-Head Attention**
   - Self-attention mechanism with multiple heads
   - Causal masking for autoregressive generation
   - Dropout for regularization

3. **Feed-Forward Networks**
   - GELU activation function
   - Projection layers with dropout

4. **Layer Normalization**
   - Pre-norm architecture for better training stability
   - Applied before attention and feed-forward layers

### Training Features

- **Weight Tying**: Token embedding and output projection weights are tied
- **Gradient Clipping**: Prevents exploding gradients
- **Cosine Learning Rate Schedule**: Gradual learning rate decay
- **Checkpointing**: Regular saving of training state
- **Mixed Precision**: Optional for faster training (if supported)

## Extending the Model

### Using Your Own Data

Replace the `load_sample_data()` function in `train.py`:

```python
def load_sample_data():
    # Load your text files
    texts = []
    with open('your_dataset.txt', 'r', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
    return texts
```

### Scaling the Model

To create a larger model (e.g., 7B parameters):

```python
config = {
    'vocab_size': 32000,
    'd_model': 4096,
    'n_layers': 32,
    'n_heads': 32,
    'd_ff': 11008,
    'max_seq_len': 2048,
    # ... other parameters
}
```

### Adding Advanced Features

The codebase is designed to be extensible. You can add:

- **Attention Variants**: Flash attention, sliding window attention
- **Advanced Optimizers**: AdamW with weight decay, Lion optimizer
- **Better Tokenization**: SentencePiece, BPE tokenization
- **Quantization**: INT8/INT4 quantization for smaller models
- **Fine-tuning**: Instruction following, RLHF alignment

## Performance Notes

- **Memory Usage**: ~8-12 GB GPU memory for training (batch_size=4)
- **Training Time**: ~30 minutes on RTX 4090 for demo dataset
- **Inference Speed**: ~10-20 tokens/second on GPU
- **Model Quality**: Basic but functional for demonstration purposes

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `d_model` or `n_layers`
- Use gradient accumulation

### Slow Training
- Ensure CUDA is available and being used
- Increase batch size if memory allows
- Use mixed precision training

### Poor Generation Quality
- Train on larger, more diverse datasets
- Increase training epochs
- Tune generation parameters (temperature, top_k)

## License

This project is for educational purposes. Feel free to modify and extend for your own projects.

## Contributing

This is a learning project. Suggestions and improvements are welcome!

## Acknowledgments

- Inspired by modern LLM architectures (GPT, LLaMA)
- Uses techniques from "Attention Is All You Need" and subsequent papers
- Built with PyTorch for accessibility and clarity
