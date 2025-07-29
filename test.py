"""
Test script to verify the Small LLM implementation works correctly
"""

import torch
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SmallLLM
from tokenizer import SimpleTokenizer

def test_tokenizer():
    """Test the tokenizer functionality"""
    print("Testing tokenizer...")
    
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Test with sample texts
    sample_texts = [
        "Hello world!",
        "This is a test sentence.",
        "Machine learning is fascinating.",
        "Deep learning models are powerful."
    ]
    
    # Train tokenizer
    tokenizer.train(sample_texts)
    
    # Test encoding/decoding
    test_text = "Hello world!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")
    
    assert isinstance(tokens, list), "Tokens should be a list"
    assert all(isinstance(t, int) for t in tokens), "All tokens should be integers"
    print("✓ Tokenizer test passed!")
    
def test_model():
    """Test the model architecture"""
    print("\nTesting model...")
    
    # Small configuration for testing
    config = {
        'vocab_size': 1000,
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 8,
        'd_ff': 1024,
        'max_seq_len': 128,
        'dropout': 0.1
    }
    
    model = SmallLLM(**config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    expected_shape = (batch_size, seq_len, config['vocab_size'])
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    
    # Test training mode with labels
    model.train()
    loss, logits = model(input_ids, labels=input_ids)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    print(f"Estimated size: {param_count * 4 / (1024**2):.1f} MB")
    
    print("✓ Model test passed!")

def test_integration():
    """Test tokenizer and model integration"""
    print("\nTesting integration...")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    sample_texts = ["Hello world!", "This is a test."]
    tokenizer.train(sample_texts)
    
    # Create model
    model = SmallLLM(
        vocab_size=1000,
        d_model=256,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_seq_len=64
    )
    
    # Test text processing
    test_text = "Hello world!"
    tokens = tokenizer.encode(test_text)
    input_ids = torch.tensor([tokens])
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    # Test that we can get next token probabilities
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6), "Probabilities should sum to 1"
    
    print("✓ Integration test passed!")

def main():
    """Run all tests"""
    print("Running Small LLM Tests")
    print("=" * 30)
    
    try:
        test_tokenizer()
        test_model()
        test_integration()
        
        print("\n" + "=" * 30)
        print("🎉 All tests passed!")
        print("\nYou can now run:")
        print("  python train.py    # to train the model")
        print("  python inference.py # to chat with the model (after training)")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
