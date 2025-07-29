"""
Enhanced Model Architecture for Vaani LLM
Incorporate successful patterns from modern LLMs (Llama, GPT-4, Nemotron, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Llama, PaLM)"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SwiGLU(nn.Module):
    """SwiGLU activation function (used in LLaMA, PaLM)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (used in Llama 2, reduces memory usage)"""
    def __init__(self, d_model, n_heads, n_kv_heads=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Grouped queries
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Repeat K, V for grouped attention
        if self.n_kv_heads != self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(out)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - used in most modern LLMs"""
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys"""
    cos = cos[:, :, :q.shape[2], :]
    sin = sin[:, :, :q.shape[2], :]
    
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with modern improvements"""
    def __init__(self, d_model, n_heads, d_ff, n_kv_heads=None, dropout=0.1):
        super().__init__()
        
        # Use RMSNorm instead of LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Use Grouped Query Attention
        self.attention = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        
        # Use SwiGLU instead of standard FFN
        self.feed_forward = SwiGLU(d_model, d_ff)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm architecture (more stable)
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

class EnhancedVaaniLLM(nn.Module):
    """Enhanced Vaani LLM with modern architectural improvements"""
    def __init__(self, vocab_size=32000, d_model=2048, n_layers=24, n_heads=32,
                 d_ff=8192, max_seq_len=4096, dropout=0.1, n_kv_heads=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerBlock(
                d_model=d_model,
                n_heads=n_heads, 
                d_ff=d_ff,
                n_kv_heads=n_kv_heads,
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings (weight sharing)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following modern practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        mask = mask.to(input_ids.device)
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer norm
        x = self.norm(x)
        
        # Output projection
        logits = self.lm_head(x)
        
        if labels is not None:
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            return loss, logits
        
        return logits
    
    def count_parameters(self):
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Enhanced generation with nucleus sampling"""
        self.eval()
        
        for _ in range(max_length):
            with torch.no_grad():
                logits = self.forward(input_ids)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < top_k_logits[-1]] = float('-inf')
                
                # Apply nucleus (top-p) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        return input_ids

def create_enhanced_model():
    """Create enhanced Vaani model with modern architecture"""
    
    # Configuration for different model sizes
    configs = {
        "small": {
            "vocab_size": 32000,
            "d_model": 1024,
            "n_layers": 12,
            "n_heads": 16,
            "n_kv_heads": 4,  # Grouped query attention
            "d_ff": 4096,
            "max_seq_len": 2048
        },
        "medium": {
            "vocab_size": 32000,
            "d_model": 1536,
            "n_layers": 18,
            "n_heads": 24,
            "n_kv_heads": 6,
            "d_ff": 6144,
            "max_seq_len": 4096
        },
        "large": {
            "vocab_size": 32000,
            "d_model": 2048,
            "n_layers": 24,
            "n_heads": 32,
            "n_kv_heads": 8,
            "d_ff": 8192,
            "max_seq_len": 4096
        }
    }
    
    # Choose configuration
    config = configs["small"]  # Change to "medium" or "large" for bigger models
    
    print("Creating Enhanced Vaani LLM...")
    print(f"Configuration: {config}")
    
    model = EnhancedVaaniLLM(**config)
    
    param_count = model.count_parameters()
    model_size_gb = param_count * 4 / (1024**3)
    
    print(f"Model parameters: {param_count:,}")
    print(f"Estimated size: {model_size_gb:.2f} GB")
    
    return model, config

def test_enhanced_model():
    """Test the enhanced model"""
    model, config = create_enhanced_model()
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    vocab_size = config["vocab_size"]
    
    # Create random input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size)
    
    # Test generation
    print("Testing generation...")
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_length=20, temperature=0.8)
    print(f"Generated sequence length: {generated.shape[1]}")
    
    print("✅ Enhanced model test passed!")
    
    return model

if __name__ == "__main__":
    # Create and test enhanced model
    model = test_enhanced_model()
    
    # Save model
    torch.save(model.state_dict(), 'enhanced_vaani_model.pth')
    print("Enhanced model saved as: enhanced_vaani_model.pth")
