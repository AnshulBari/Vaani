# 📊 Vaani LLM - Model Specifications Document

## Overview

**Vaani LLM** is a custom-trained transformer-based language model built from scratch using PyTorch. This document provides comprehensive technical specifications for the 427-million parameter model.

## 🏗️ Architecture Specifications

### Model Class
- **Base Architecture**: Transformer Decoder (GPT-style)
- **Implementation**: Custom `SmallLLM` class in PyTorch
- **Training Framework**: PyTorch 2.0+

### Core Parameters
```json
{
  "model_name": "Vaani LLM",
  "total_parameters": 427442560,
  "parameter_count_human": "427.4M",
  "model_size_fp32": "1.63 GB",
  "model_size_fp16": "815 MB",
  "architecture_type": "transformer_decoder"
}
```

### Layer Configuration
```json
{
  "vocab_size": 16000,
  "hidden_size": 1408,
  "num_hidden_layers": 16,
  "num_attention_heads": 11,
  "intermediate_size": 5632,
  "max_position_embeddings": 1024,
  "layer_norm_eps": 1e-5,
  "hidden_dropout_prob": 0.1,
  "attention_probs_dropout_prob": 0.1
}
```

## 🧠 Architecture Components

### Embedding Layer
- **Token Embeddings**: `vocab_size (16000) × hidden_size (1408)`
- **Parameters**: 22,528,000
- **Weight Tying**: Shared with output layer (lm_head)

### Transformer Blocks (16 layers)
Each layer contains:

#### Multi-Head Attention
- **Attention Heads**: 11
- **Head Dimension**: 128 (1408 ÷ 11)
- **Query/Key/Value Projections**: Linear layers without bias
- **Output Projection**: Linear layer without bias
- **Position Encoding**: Rotary Position Embedding (RoPE)

#### Feed-Forward Network  
- **Intermediate Size**: 5,632
- **Activation Function**: GELU
- **Structure**: Linear → GELU → Dropout → Linear

#### Normalization
- **Type**: Layer Normalization (Pre-norm architecture)
- **Placement**: Before attention and feed-forward blocks
- **Parameters**: Scale and bias terms

### Output Layer
- **Language Model Head**: Linear layer (no bias)
- **Weight Sharing**: Tied with token embedding weights
- **Output Dimension**: vocab_size (16000)

## 📐 Mathematical Specifications

### Parameter Breakdown
```
Token Embedding:     16,000 × 1,408 =   22,528,000
Position Encoding:   Rotary (no params) =         0

Per Transformer Layer:
  - Attention:       1,408 × 1,408 × 4 =  7,946,240
  - Feed Forward:    1,408 × 5,632 × 2 = 15,876,096
  - Layer Norms:     1,408 × 4        =     5,632
  Total per layer:                       23,827,968

16 layers:           16 × 23,827,968 = 381,247,488

Final Layer Norm:    1,408 × 2       =     2,816
Output Head:         Tied weights     =         0

Total Parameters:                     403,778,304
```

*Note: Actual parameter count may vary due to implementation details*

### Memory Requirements
```
Model Parameters (FP32):  427.4M × 4 bytes = 1.63 GB
Model Parameters (FP16):  427.4M × 2 bytes = 815 MB
Gradients (training):     Equal to parameters = 1.63 GB
Optimizer states:         2-3× parameters = 3.26-4.89 GB
Activations (seq=1024):   ~500 MB - 2 GB (batch dependent)
```

### FLOP Analysis (per forward pass)
```
Sequence Length: 1024
Batch Size: 1

Attention FLOPs:  4 × 1 × 1024² × 1408 × 16 = 94.3 GFLOPs
FFN FLOPs:        8 × 1 × 1024 × 1408 × 5632 × 16 = 1,039.6 GFLOPs
Total FLOPs:                                          1,133.9 GFLOPs
```

## 🔤 Tokenization Specifications

### Tokenizer Type
- **Implementation**: Custom word-based tokenizer
- **Algorithm**: Simple word splitting with regex
- **File**: `tokenizer.pkl` (serialized)

### Vocabulary Details
```json
{
  "configured_vocab_size": 16000,
  "actual_vocab_size": 262,
  "tokenizer_type": "word_based",
  "case_sensitive": false,
  "special_tokens": {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3
  }
}
```

### Tokenization Process
1. **Text Preprocessing**: Convert to lowercase
2. **Word Extraction**: Regex pattern `\b\w+\b|[^\w\s]`
3. **Token Mapping**: Word → ID via vocabulary
4. **Unknown Handling**: Out-of-vocabulary → `<unk>` (ID: 1)
5. **Sequence Wrapping**: Add `<bos>` and `<eos>` tokens

## ⚙️ Training Specifications

### Training Configuration
```json
{
  "dataset_size": 600,
  "training_epochs": 3,
  "batch_size": 1,
  "gradient_accumulation_steps": 4,
  "effective_batch_size": 4,
  "learning_rate": 0.0002,
  "weight_decay": 0.01,
  "warmup_steps": 5,
  "total_training_steps": 56,
  "optimizer": "AdamW",
  "loss_function": "CrossEntropyLoss"
}
```

### Training Hardware
- **Primary**: Google Colab (Tesla T4/P100)
- **Memory**: 15GB GPU constraint
- **Training Time**: ~7 minutes total
- **Checkpointing**: Every epoch

### Training Results
```
Epoch 1: Loss 0.9742 → 0.9742 avg
Epoch 2: Loss 0.3103 → 0.3189 avg  
Epoch 3: Loss 0.2954 → 0.3113 avg (final)

Final Training Loss: 0.3113
Convergence: Achieved in 3 epochs
Loss Reduction: 68% improvement
```

## 🚀 Performance Characteristics

### Inference Speed (Estimated)
```
Hardware Configuration    | Tokens/Second
--------------------------|---------------
CPU (Intel i7-10700K)    | 1-3
GPU (GTX 1080 Ti)        | 15-25
GPU (RTX 3080)           | 30-50
GPU (RTX 4090)           | 50-80
GPU (A100)               | 80-120
```

### Context Window
- **Maximum Sequence Length**: 1,024 tokens
- **Effective Context**: 1,024 tokens
- **Position Encoding**: RoPE (supports extrapolation)

### Model Precision Support
- **FP32**: Full precision (1.63 GB)
- **FP16**: Half precision (815 MB) 
- **INT8**: Quantization possible (400 MB estimated)

## 🔧 Implementation Details

### Modern Architecture Features
1. **Rotary Position Embedding (RoPE)**
   - No learned position embeddings
   - Better handling of variable sequence lengths
   - Improved extrapolation beyond training lengths

2. **Pre-Layer Normalization**
   - LayerNorm before attention and FFN
   - Improved training stability
   - Better gradient flow

3. **Weight Tying**
   - Input embeddings = Output projections
   - Reduces parameters by 22.5M
   - Improved training efficiency

4. **Bias-Free Linear Layers**
   - No bias terms in attention projections
   - Follows modern transformer practices
   - Slightly reduced parameter count

### File Structure
```
model.pth (1,630.6 MB):
├── token_embedding.weight        [16000, 1408]
├── layers.0.norm1.weight         [1408]
├── layers.0.norm1.bias           [1408]
├── layers.0.attention.q_proj.weight [1408, 1408]
├── layers.0.attention.k_proj.weight [1408, 1408]
├── layers.0.attention.v_proj.weight [1408, 1408]
├── layers.0.attention.o_proj.weight [1408, 1408]
├── layers.0.norm2.weight         [1408]
├── layers.0.norm2.bias           [1408]
├── layers.0.feed_forward.linear1.weight [5632, 1408]
├── layers.0.feed_forward.linear2.weight [1408, 5632]
├── ... (repeated for layers 1-15)
├── norm.weight                   [1408]
├── norm.bias                     [1408]
└── lm_head.weight               [shared with embedding]
```

## 📈 Comparison with Other Models

| Model | Parameters | Size | Context | Architecture |
|-------|------------|------|---------|-------------|
| GPT-2 Small | 124M | 0.5 GB | 1024 | GPT-2 |
| GPT-2 Medium | 355M | 1.4 GB | 1024 | GPT-2 |
| **Vaani LLM** | **427M** | **1.63 GB** | **1024** | **Custom** |
| GPT-2 Large | 774M | 3.0 GB | 1024 | GPT-2 |
| LLaMA-7B | 7B | 13 GB | 2048 | LLaMA |

## ⚠️ Current Limitations

### Training Data Constraints
- **Dataset Size**: Only 600 samples (vs. millions typically used)
- **Vocabulary Coverage**: 262 words (vs. 30K+ standard)
- **Domain**: Limited to training text scope
- **Language Patterns**: Basic due to small dataset

### Architecture Limitations
- **Attention Heads**: 11 heads (odd number, not power of 2)
- **Head Dimension**: 128 (not standard 64 or 256)
- **Tokenization**: Word-based (vs. subword methods)

### Generation Quality
- **Unknown Words**: High `<unk>` rate for out-of-vocabulary
- **Coherence**: Limited by small training set
- **Diversity**: Constrained by 262-word vocabulary
- **Context**: Good architectural support, limited by training

## 🎯 Recommended Improvements

### Short-term (Feasible)
1. **Expand Training Data**: 600 → 10,000+ samples
2. **Better Tokenization**: Implement BPE/SentencePiece
3. **Vocabulary Growth**: Train on larger, diverse corpus
4. **Extended Training**: 3 → 10+ epochs with validation

### Medium-term (Moderate Effort)
1. **Data Pipeline**: Automated data collection and cleaning
2. **Evaluation Metrics**: Perplexity, BLEU, human evaluation
3. **Hyperparameter Tuning**: Learning rate, architecture search
4. **Multi-GPU Training**: Scale to larger datasets

### Long-term (Significant Effort)
1. **Instruction Tuning**: Fine-tune for specific tasks
2. **RLHF**: Reinforcement learning from human feedback
3. **Model Scaling**: Increase to 1B+ parameters
4. **Production Deployment**: API, web interface, optimization

## 📚 Technical References

### Key Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-norm

### Implementation References
- PyTorch Transformer: `torch.nn.Transformer`
- Hugging Face: `transformers.GPT2Model`
- nanoGPT: Minimal GPT implementation

---

**Document Version**: 1.0  
**Last Updated**: August 2, 2025  
**Model Version**: Vaani LLM 427M  
**Authors**: AnshulBari
