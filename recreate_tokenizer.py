#!/usr/bin/env python3
"""
Script to recreate the tokenizer.pkl file for Vaani LLM
This will create a basic tokenizer with the correct vocabulary size
"""

import torch
import pickle
from tokenizer import SimpleTokenizer
from model import SmallLLM
import json

def get_actual_vocab_size():
    """Get the actual vocabulary size from the saved model"""
    try:
        # Load the model to get the actual vocabulary size
        model = torch.load('model.pth', map_location='cpu')
        
        # Check if it's a state dict or the full model
        if isinstance(model, dict):
            # It's a state dict - look for embedding layer
            for key, value in model.items():
                if 'embedding' in key.lower() and 'weight' in key:
                    vocab_size = value.shape[0]
                    print(f"Found vocabulary size from model state dict: {vocab_size}")
                    return vocab_size
        else:
            # It's the full model
            if hasattr(model, 'embedding'):
                vocab_size = model.embedding.weight.shape[0]
                print(f"Found vocabulary size from model object: {vocab_size}")
                return vocab_size
        
        # Fallback to config file
        with open('config.json', 'r') as f:
            config = json.load(f)
            vocab_size = config.get('vocab_size', 16000)
            print(f"Using vocabulary size from config: {vocab_size}")
            return vocab_size
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using default vocabulary size from config")
        with open('config.json', 'r') as f:
            config = json.load(f)
            return config.get('vocab_size', 16000)

def create_basic_vocabulary(vocab_size):
    """Create a basic vocabulary for the tokenizer"""
    # Start with special tokens
    vocabulary = ['<pad>', '<unk>', '<bos>', '<eos>']
    
    # Add common English words and characters
    common_words = [
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
        'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if',
        'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
        'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
        'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
        'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want',
        'because', 'any', 'these', 'give', 'day', 'most', 'us'
    ]
    
    # Add single characters and punctuation
    for char in 'abcdefghijklmnopqrstuvwxyz0123456789.,!?;:"()[]{}/-':
        vocabulary.append(char)
    
    # Add more common words until we reach the desired vocabulary size
    vocabulary.extend(common_words)
    
    # Fill remaining slots with generated tokens
    while len(vocabulary) < vocab_size:
        vocabulary.append(f'<token_{len(vocabulary)}>')
    
    return vocabulary[:vocab_size]

def recreate_tokenizer():
    """Recreate the tokenizer with proper vocabulary"""
    print("Recreating tokenizer...")
    
    # Get the actual vocabulary size from the model
    vocab_size = get_actual_vocab_size()
    print(f"Using vocabulary size: {vocab_size}")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    
    # Create basic vocabulary
    vocabulary = create_basic_vocabulary(vocab_size)
    
    # Manually set up the tokenizer mappings
    tokenizer.word_to_id = {word: i for i, word in enumerate(vocabulary)}
    tokenizer.id_to_word = {i: word for word, i in tokenizer.word_to_id.items()}
    
    # Update special tokens to match the vocabulary
    tokenizer.special_tokens = {
        '<pad>': 0,
        '<unk>': 1,
        '<bos>': 2,
        '<eos>': 3,
    }
    
    print(f"Created tokenizer with {len(tokenizer.word_to_id)} tokens")
    
    # Save the tokenizer
    tokenizer.save('tokenizer.pkl')
    print("Tokenizer saved as tokenizer.pkl")
    
    # Test the tokenizer
    test_text = "Hello, this is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nTest encoding/decoding:")
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    return tokenizer

if __name__ == "__main__":
    recreate_tokenizer()
    print("\nTokenizer recreation complete!")
