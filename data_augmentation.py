"""
Data Augmentation for Vaani LLM
Use larger models (GPT-3.5, GPT-4, Claude, etc.) to generate high-quality training data
"""

import torch
import json
import time
from typing import List, Dict
import requests
from tqdm import tqdm

class DataAugmenter:
    def __init__(self):
        """Initialize data augmentation system"""
        # You can add API keys here or use environment variables
        pass
    
    def generate_with_huggingface(self, prompts: List[str], model_name="microsoft/DialoGPT-medium") -> List[str]:
        """Generate training data using HuggingFace models"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        generated_texts = []
        
        for prompt in tqdm(prompts, desc="Generating with HuggingFace"):
            try:
                inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 100,
                        temperature=0.8,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the original prompt
                generated_text = generated_text[len(prompt):].strip()
                generated_texts.append(generated_text)
                
            except Exception as e:
                print(f"Error generating text: {e}")
                generated_texts.append("")
        
        return generated_texts
    
    def create_synthetic_qa_dataset(self, topics: List[str]) -> List[Dict]:
        """Create Q&A training data"""
        qa_pairs = []
        
        question_templates = [
            "What is {topic}?",
            "How does {topic} work?", 
            "Why is {topic} important?",
            "What are the benefits of {topic}?",
            "Can you explain {topic} in simple terms?",
            "What are the main features of {topic}?",
            "How can {topic} be used?",
            "What should I know about {topic}?"
        ]
        
        for topic in topics:
            for template in question_templates:
                question = template.format(topic=topic)
                
                # Generate synthetic answer
                answer = self._generate_synthetic_answer(topic, question)
                
                qa_pairs.append({
                    "question": question,
                    "answer": answer,
                    "topic": topic
                })
        
        return qa_pairs
    
    def _generate_synthetic_answer(self, topic: str, question: str) -> str:
        """Generate synthetic answers for training"""
        # Simple rule-based generation (you can replace with model-based generation)
        topic_info = {
            "artificial intelligence": "Artificial intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence. AI systems use algorithms and data to learn patterns, make decisions, and solve problems. Applications include image recognition, natural language processing, and autonomous vehicles.",
            
            "machine learning": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses statistical techniques to identify patterns in data and make predictions or decisions.",
            
            "deep learning": "Deep learning is a type of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns. It's particularly effective for tasks like image recognition, speech processing, and natural language understanding.",
            
            "neural networks": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections. They can learn to recognize patterns and make predictions through training.",
            
            "transformers": "Transformers are a type of neural network architecture that uses attention mechanisms to process sequential data. They've revolutionized natural language processing and are the foundation of models like GPT and BERT.",
            
            "attention mechanism": "Attention mechanisms allow neural networks to focus on specific parts of the input when making predictions. This helps models handle long sequences and understand relationships between different parts of the data.",
            
            "natural language processing": "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language. Applications include translation, sentiment analysis, and chatbots.",
            
            "computer vision": "Computer vision is a field of AI that enables machines to interpret and understand visual information from the world. It involves processing and analyzing digital images and videos to extract meaningful information."
        }
        
        # Try to find specific information
        for key, value in topic_info.items():
            if key in topic.lower():
                return value
        
        # Generic response
        return f"{topic.title()} is an important concept in technology and science. It involves various principles and applications that are relevant to understanding modern systems and processes. Further research and study can provide deeper insights into this topic."
    
    def expand_existing_dataset(self, texts: List[str]) -> List[str]:
        """Expand existing dataset through paraphrasing and variation"""
        expanded_texts = []
        
        # Add original texts
        expanded_texts.extend(texts)
        
        # Create variations
        for text in texts:
            # Simple paraphrasing techniques
            sentences = text.split('. ')
            
            if len(sentences) > 1:
                # Reorder sentences
                import random
                random.shuffle(sentences)
                reordered = '. '.join(sentences)
                expanded_texts.append(reordered)
                
                # Take subsets
                if len(sentences) > 2:
                    subset = '. '.join(sentences[:len(sentences)//2])
                    expanded_texts.append(subset + '.')
            
            # Create question-answer pairs
            for sentence in sentences[:2]:  # Use first 2 sentences
                if len(sentence.strip()) > 10:
                    question = f"What can you tell me about: {sentence.strip()[:50]}...?"
                    expanded_texts.append(f"Question: {question} Answer: {sentence.strip()}")
        
        return expanded_texts
    
    def create_instruction_dataset(self) -> List[Dict]:
        """Create instruction-following dataset"""
        instructions = [
            {
                "instruction": "Explain the concept of machine learning",
                "input": "",
                "output": "Machine learning is a branch of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It uses algorithms to identify patterns in data and make predictions or classifications on new, unseen data."
            },
            {
                "instruction": "Describe the difference between supervised and unsupervised learning",
                "input": "",
                "output": "Supervised learning uses labeled training data to learn relationships between inputs and outputs, like classifying emails as spam or not spam. Unsupervised learning finds hidden patterns in data without labels, such as grouping customers by purchasing behavior."
            },
            {
                "instruction": "What are neural networks?",
                "input": "",
                "output": "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) organized in layers that process information through weighted connections. They can learn to recognize complex patterns through training."
            },
            {
                "instruction": "Explain the attention mechanism",
                "input": "",
                "output": "The attention mechanism allows neural networks to focus on specific parts of the input when making predictions. Instead of processing all input equally, attention helps models identify which information is most relevant for the current task, improving performance on long sequences."
            },
            {
                "instruction": "Generate a simple explanation for a complex topic",
                "input": "quantum computing",
                "output": "Quantum computing uses the strange properties of quantum physics to process information differently than regular computers. While normal computers use bits (0 or 1), quantum computers use quantum bits that can be both 0 and 1 at the same time, potentially solving certain problems much faster."
            }
        ]
        
        return instructions

def generate_augmented_dataset():
    """Main function to generate augmented training dataset"""
    
    augmenter = DataAugmenter()
    
    # Define topics for data generation
    topics = [
        "artificial intelligence",
        "machine learning", 
        "deep learning",
        "neural networks",
        "transformers",
        "attention mechanism",
        "natural language processing",
        "computer vision",
        "robotics",
        "data science",
        "algorithms",
        "programming",
        "technology",
        "innovation",
        "future trends"
    ]
    
    print("Generating augmented training dataset...")
    
    # Create Q&A dataset
    qa_dataset = augmenter.create_synthetic_qa_dataset(topics)
    print(f"Generated {len(qa_dataset)} Q&A pairs")
    
    # Create instruction dataset
    instruction_dataset = augmenter.create_instruction_dataset()
    print(f"Generated {len(instruction_dataset)} instruction examples")
    
    # Load existing data and expand it
    try:
        from train import create_sample_dataset
        existing_texts = create_sample_dataset()
        expanded_texts = augmenter.expand_existing_dataset(existing_texts)
        print(f"Expanded to {len(expanded_texts)} text samples")
    except:
        expanded_texts = []
    
    # Combine all data
    all_training_data = []
    
    # Add Q&A data
    for qa in qa_dataset:
        all_training_data.append(f"Question: {qa['question']} Answer: {qa['answer']}")
    
    # Add instruction data
    for inst in instruction_dataset:
        if inst['input']:
            all_training_data.append(f"Instruction: {inst['instruction']} Input: {inst['input']} Output: {inst['output']}")
        else:
            all_training_data.append(f"Instruction: {inst['instruction']} Output: {inst['output']}")
    
    # Add expanded texts
    all_training_data.extend(expanded_texts)
    
    # Save augmented dataset
    output_file = 'augmented_training_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'qa_data': qa_dataset,
            'instruction_data': instruction_dataset, 
            'expanded_texts': expanded_texts,
            'combined_training_data': all_training_data,
            'total_samples': len(all_training_data)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Augmented dataset saved to {output_file}")
    print(f"Total training samples: {len(all_training_data)}")
    
    # Also save as plain text for easy use
    with open('augmented_training_texts.txt', 'w', encoding='utf-8') as f:
        for text in all_training_data:
            f.write(text + '\n\n')
    
    print("Plain text version saved to augmented_training_texts.txt")
    
    return all_training_data

if __name__ == "__main__":
    generate_augmented_dataset()
