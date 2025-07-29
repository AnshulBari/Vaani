#!/usr/bin/env python3
"""
Vaani LLM Demonstration Script
Shows model capabilities and suggests better prompts based on vocabulary
"""

import torch
import pickle
from local_inference import load_model_and_tokenizer, generate_text

def explore_vocabulary(tokenizer):
    """Explore the model's vocabulary to suggest good prompts"""
    word_to_id = tokenizer.get('word_to_id', {})
    
    # Get meaningful words (exclude special tokens)
    meaningful_words = []
    for word, token_id in word_to_id.items():
        if not word.startswith('<') and word not in ['.', ',', '-', '!', '?']:
            meaningful_words.append(word)
    
    print("🔤 Vaani's Vocabulary Preview:")
    print("=" * 50)
    
    # Show categories of words the model knows
    categories = {
        'Technology': ['artificial', 'intelligence', 'machine', 'learning', 'computer', 'data', 'algorithms', 'neural', 'networks'],
        'Science': ['science', 'universe', 'physics', 'quantum', 'energy', 'matter', 'research'],
        'General': ['the', 'and', 'is', 'to', 'of', 'in', 'that', 'for', 'with', 'are']
    }
    
    for category, sample_words in categories.items():
        known_words = [word for word in sample_words if word in word_to_id]
        if known_words:
            print(f"\n{category}: {', '.join(known_words[:10])}")
    
    print(f"\nTotal vocabulary: {len(meaningful_words)} meaningful words")
    print(f"Sample words: {', '.join(meaningful_words[:20])}")
    
    return meaningful_words

def suggest_prompts(meaningful_words):
    """Suggest prompts that work well with the model's vocabulary"""
    
    # Create prompts using words the model actually knows
    good_prompts = [
        "artificial intelligence",
        "machine learning",
        "the future",
        "science and technology", 
        "computer systems",
        "data processing",
        "neural networks",
        "quantum physics",
        "human knowledge"
    ]
    
    print("\n💡 Suggested Prompts (using known vocabulary):")
    print("=" * 50)
    
    for i, prompt in enumerate(good_prompts, 1):
        print(f"{i}. \"{prompt}\"")
    
    return good_prompts

def demo_generation(model, tokenizer, device):
    """Demonstrate text generation with various prompts"""
    meaningful_words = explore_vocabulary(tokenizer)
    suggested_prompts = suggest_prompts(meaningful_words)
    
    print("\n🎭 Generation Demo:")
    print("=" * 50)
    
    # Test a few suggested prompts automatically
    test_prompts = suggested_prompts[:3]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: \"{prompt}\"")
        result = generate_text(model, tokenizer, prompt, max_length=20, temperature=0.7, top_k=30, device=device)
        print(f"🤖 Vaani: {result}")
        print("-" * 40)

def interactive_mode(model, tokenizer, device):
    """Interactive chat mode with vocabulary guidance"""
    word_to_id = tokenizer.get('word_to_id', {})
    
    print("\n💬 Interactive Mode:")
    print("=" * 50)
    print("💡 Tips for better results:")
    print("- Use simple words from the vocabulary shown above")
    print("- Try technology/science related topics")
    print("- Keep prompts short (2-4 words)")
    print("- Type 'vocab' to see vocabulary again")
    print("- Type 'quit' to exit")
    
    while True:
        user_input = input("\n🗣️  You: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'vocab':
            explore_vocabulary(tokenizer)
            continue
        elif not user_input:
            continue
        
        # Check how many words are in vocabulary
        words = user_input.lower().split()
        unknown_words = [word for word in words if word not in word_to_id]
        
        if unknown_words:
            print(f"⚠️  Unknown words: {', '.join(unknown_words)}")
            print("💡 Try using simpler or more common words")
        
        result = generate_text(model, tokenizer, user_input, max_length=25, temperature=0.8, top_k=40, device=device)
        print(f"🤖 Vaani: {result}")

def main():
    """Main demonstration function"""
    print("🤖 Vaani LLM Demonstration")
    print("=" * 60)
    
    # Load model
    print("📂 Loading Vaani model...")
    model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        print("❌ Failed to load model. Please ensure model files exist.")
        return
    
    print(f"✅ Model loaded successfully on {device}")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run demonstration
    demo_generation(model, tokenizer, device)
    
    # Interactive mode
    interactive_mode(model, tokenizer, device)

if __name__ == "__main__":
    main()
