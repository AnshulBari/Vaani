#!/usr/bin/env python3
"""
Vaani LLM Analysis Script
Analyzes what the model has learned and demonstrates its capabilities
"""

import torch
import pickle
from local_inference import load_model_and_tokenizer, generate_text

def analyze_model_knowledge(model, tokenizer, device):
    """Analyze what patterns the model has learned"""
    word_to_id = tokenizer.get('word_to_id', {})
    id_to_word = tokenizer.get('id_to_word', {})
    
    print("🔍 Model Knowledge Analysis")
    print("=" * 50)
    
    # Test specific word associations
    test_cases = [
        ("artificial", "Should relate to technology/intelligence"),
        ("machine", "Should relate to learning/technology"),
        ("the", "Common article, should lead to nouns"),
        ("science", "Should relate to research/knowledge"),
        ("quantum", "Should relate to physics/science"),
        ("neural", "Should relate to networks/learning"),
        ("future", "Should relate to predictions/time"),
    ]
    
    print("🧠 Word Association Analysis:")
    print("-" * 30)
    
    for word, expected in test_cases:
        if word in word_to_id:
            # Generate continuation for single word
            result = generate_text(model, tokenizer, word, max_length=10, temperature=0.3, top_k=20, device=device)
            print(f"'{word}' → {result}")
            print(f"   Expected: {expected}")
            print()
    
def demonstrate_capabilities(model, tokenizer, device):
    """Demonstrate what the model can and cannot do"""
    print("\n✅ What Vaani CAN do:")
    print("=" * 30)
    
    capabilities = [
        ("Complete technical terms", "artificial intelligence"),
        ("Form basic sentences", "the future of"),
        ("Use common words", "science and"),
        ("Handle punctuation", "learning."),
    ]
    
    for description, prompt in capabilities:
        result = generate_text(model, tokenizer, prompt, max_length=15, temperature=0.5, device=device)
        print(f"✅ {description}:")
        print(f"   Input: '{prompt}' → Output: '{result}'")
        print()
    
    print("❌ What Vaani CANNOT do well:")
    print("=" * 30)
    
    limitations = [
        ("Handle unknown words", "How are you doing today?"),
        ("Answer questions", "What is your name?"),
        ("Follow instructions", "Write a poem about cats"),
        ("Maintain long context", "Tell me a long story about adventure"),
    ]
    
    for description, prompt in limitations:
        result = generate_text(model, tokenizer, prompt, max_length=15, temperature=0.5, device=device)
        print(f"❌ {description}:")
        print(f"   Input: '{prompt}' → Output: '{result}'")
        print()

def vocabulary_analysis(tokenizer):
    """Analyze the vocabulary composition"""
    word_to_id = tokenizer.get('word_to_id', {})
    
    print("\n📊 Vocabulary Analysis:")
    print("=" * 30)
    
    # Categorize words
    categories = {
        'Special Tokens': [],
        'Punctuation': [],
        'Common Words': [],
        'Technology': [],
        'Science': [],
        'Other': []
    }
    
    tech_words = ['artificial', 'intelligence', 'machine', 'learning', 'computer', 'data', 'algorithms', 'neural', 'networks', 'processing', 'technological']
    science_words = ['science', 'universe', 'physics', 'quantum', 'energy', 'matter', 'research', 'scientific', 'study', 'nature']
    common_words = ['the', 'and', 'is', 'to', 'of', 'in', 'that', 'for', 'with', 'are', 'this', 'as', 'by', 'from', 'they', 'we', 'or', 'an', 'be', 'at']
    
    for word in word_to_id.keys():
        if word.startswith('<') and word.endswith('>'):
            categories['Special Tokens'].append(word)
        elif word in ['.', ',', '!', '?', ';', ':', '-']:
            categories['Punctuation'].append(word)
        elif word in tech_words:
            categories['Technology'].append(word)
        elif word in science_words:
            categories['Science'].append(word)
        elif word in common_words:
            categories['Common Words'].append(word)
        else:
            categories['Other'].append(word)
    
    for category, words in categories.items():
        print(f"{category}: {len(words)} words")
        if len(words) <= 10:
            print(f"   {', '.join(words)}")
        else:
            print(f"   {', '.join(words[:10])}... (+{len(words)-10} more)")
        print()

def training_quality_assessment(model, tokenizer, device):
    """Assess the quality of training"""
    print("\n🎯 Training Quality Assessment:")
    print("=" * 35)
    
    # Test consistency
    test_prompt = "artificial intelligence"
    print(f"🔄 Consistency Test (same input multiple times):")
    print(f"Input: '{test_prompt}'")
    print()
    
    for i in range(3):
        result = generate_text(model, tokenizer, test_prompt, max_length=10, temperature=0.1, device=device)
        print(f"   Run {i+1}: {result}")
    
    print(f"\n🌡️ Temperature Test (creativity levels):")
    temps = [0.1, 0.5, 1.0]
    for temp in temps:
        result = generate_text(model, tokenizer, "the future", max_length=10, temperature=temp, device=device)
        print(f"   Temp {temp}: {result}")

def main():
    """Main analysis function"""
    print("🔬 Vaani LLM Deep Analysis")
    print("=" * 60)
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        print("❌ Failed to load model.")
        return
    
    print(f"✅ Model loaded on {device}")
    
    # Run analyses
    vocabulary_analysis(tokenizer)
    analyze_model_knowledge(model, tokenizer, device)
    demonstrate_capabilities(model, tokenizer, device)
    training_quality_assessment(model, tokenizer, device)
    
    print("\n📋 Summary:")
    print("=" * 20)
    print("• Vaani has learned basic language patterns")
    print("• Works best with technology/science topics")
    print("• Limited by small vocabulary (262 words)")
    print("• Shows consistent but simple text generation")
    print("• Ready for scaling with more data!")

if __name__ == "__main__":
    main()
