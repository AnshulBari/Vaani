import random
import json
import torch

def create_comprehensive_dataset():
    """Create a comprehensive training dataset with diverse topics"""
    
    training_texts = [
        # Technology & AI
        """Artificial intelligence has transformed numerous industries through machine learning algorithms 
        that can process vast datasets and identify complex patterns. Deep learning models, particularly 
        transformer architectures, have revolutionized natural language processing by enabling computers 
        to understand and generate human-like text. These models learn through self-supervised training 
        on large text corpora, developing rich representations of language structure and semantics.
        The attention mechanism allows models to focus on relevant parts of input sequences, enabling 
        better understanding of context and relationships between words. Neural networks consist of 
        interconnected layers that transform input data through weighted connections and activation functions.""",
        
        # Science & Research
        """Scientific research follows the scientific method, involving hypothesis formation, experimental 
        design, data collection, and statistical analysis. Peer review ensures research quality and 
        reproducibility. Modern science increasingly relies on computational methods, big data analytics, 
        and interdisciplinary collaboration to tackle complex problems ranging from climate change to 
        disease prevention and space exploration. Researchers use various methodologies including 
        controlled experiments, observational studies, and computational simulations to understand 
        natural phenomena and test theoretical predictions.""",
        
        # Literature & Philosophy
        """Literature explores the human condition through narrative, character development, and thematic 
        exploration. Authors use literary devices such as metaphor, symbolism, and irony to convey 
        deeper meanings. Philosophy examines fundamental questions about existence, consciousness, 
        ethics, and knowledge. These disciplines help us understand ourselves and our place in the world.
        Critical analysis involves examining texts for underlying themes, cultural context, and 
        philosophical implications. Writers craft stories that reflect societal values and challenge 
        conventional thinking through innovative narrative techniques.""",
        
        # History & Culture
        """Historical analysis reveals patterns in human behavior, social development, and cultural evolution. 
        Civilizations rise and fall through complex interactions of political, economic, and environmental 
        factors. Understanding history helps us learn from past mistakes and successes, informing better 
        decision-making for current challenges and future planning. Cultural diversity enriches human 
        experience through different perspectives, traditions, and ways of understanding the world.
        Archaeological evidence provides insights into ancient societies and their technological achievements.""",
        
        # Mathematics & Logic
        """Mathematics provides the foundation for scientific understanding and technological advancement. 
        From basic arithmetic to advanced calculus and abstract algebra, mathematical concepts enable 
        precise reasoning and problem-solving. Logic and proof techniques ensure the validity of 
        mathematical arguments and form the basis for computer science and artificial intelligence.
        Statistical methods help analyze data and quantify uncertainty in real-world phenomena.
        Geometric principles describe spatial relationships and form the basis for engineering applications.""",
        
        # Business & Economics
        """Economic systems coordinate the production, distribution, and consumption of goods and services.
        Market mechanisms involve supply and demand dynamics that determine prices and resource allocation.
        Business strategies focus on creating value for customers while maintaining competitive advantages.
        Financial markets facilitate capital flows and enable investment in productive activities.
        Entrepreneurship drives innovation and economic growth through the creation of new ventures.""",
        
        # Health & Medicine
        """Medical research advances our understanding of human biology and disease mechanisms.
        Clinical trials test the safety and efficacy of new treatments and therapies.
        Public health initiatives focus on disease prevention and health promotion at population levels.
        Diagnostic techniques enable early detection and accurate assessment of medical conditions.
        Healthcare systems must balance quality, accessibility, and cost-effectiveness.""",
        
        # Environment & Nature
        """Ecological systems involve complex interactions between organisms and their environments.
        Climate change affects global weather patterns and ecosystem stability.
        Conservation efforts aim to protect biodiversity and preserve natural habitats.
        Renewable energy technologies offer sustainable alternatives to fossil fuels.
        Environmental monitoring helps track changes in air quality, water resources, and wildlife populations.""",
        
        # Education & Learning
        """Educational systems transmit knowledge, skills, and cultural values across generations.
        Pedagogical approaches vary based on learning objectives and student characteristics.
        Technology integration transforms classroom experiences and enables distance learning.
        Assessment methods measure student progress and inform instructional decisions.
        Lifelong learning becomes increasingly important in rapidly changing knowledge economies.""",
        
        # Art & Creativity
        """Artistic expression encompasses visual arts, music, literature, and performance.
        Creative processes involve imagination, experimentation, and technical skill development.
        Cultural movements reflect societal changes and challenge artistic conventions.
        Aesthetic theories explore the nature of beauty and artistic appreciation.
        Digital media expands possibilities for creative expression and audience engagement."""
    ]
    
    # Create variations and expand dataset
    expanded_texts = []
    
    # Add original texts
    for text in training_texts:
        expanded_texts.append(text.strip())
    
    # Create sentence-level variations
    for text in training_texts:
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
        
        # Create different permutations
        for i in range(15):  # More variations
            shuffled = sentences.copy()
            random.shuffle(shuffled)
            expanded_texts.append(' '.join(shuffled))
            
            # Also create partial combinations
            if len(sentences) >= 3:
                subset_size = random.randint(3, len(sentences))
                subset = random.sample(sentences, subset_size)
                expanded_texts.append(' '.join(subset))
    
    # Add dialogue and conversational examples
    dialogue_examples = [
        """Question: What is machine learning? Answer: Machine learning is a subset of artificial intelligence 
        that enables computers to learn and make decisions from data without being explicitly programmed 
        for every task. It involves algorithms that can identify patterns and improve performance over time.""",
        
        """Question: How do neural networks work? Answer: Neural networks consist of interconnected nodes 
        called neurons that process information through weighted connections. Input data flows through 
        layers of neurons, with each layer transforming the data until producing an output.""",
        
        """Question: What is the scientific method? Answer: The scientific method is a systematic approach 
        to understanding the natural world through observation, hypothesis formation, experimentation, 
        and analysis. It ensures reliable and reproducible knowledge generation."""
    ]
    
    expanded_texts.extend(dialogue_examples * 30)  # Add conversational data
    
    # Multiply dataset for substantial training volume
    final_dataset = expanded_texts * 100  # Large training corpus
    
    print(f"Created comprehensive dataset with {len(final_dataset)} samples")
    print(f"Estimated total tokens: {sum(len(text.split()) for text in final_dataset):,}")
    
    return final_dataset

def get_training_config():
    """Optimized training configuration for SmallLLM model"""
    
    return {
        # Model Architecture (targeting ~4GB model)
        'vocab_size': 16000,      # Balanced vocabulary size
        'd_model': 1024,          # Model dimension
        'n_layers': 18,           # Good depth for performance
        'n_heads': 16,            # Multi-head attention
        'd_ff': 4096,             # Feed-forward dimension
        'max_seq_len': 512,       # Sequence length
        'dropout': 0.1,
        
        # Training Hyperparameters
        'batch_size': 4,
        'gradient_accumulation_steps': 8,  # Effective batch size = 32
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.95,
        'eps': 1e-8,
        
        # Training Schedule
        'epochs': 3,
        'warmup_steps': 1000,
        'save_every': 500,
        
        # Optimization Features
        'gradient_clipping': 1.0,
        'mixed_precision': True,
        
        # Data Processing
        'overlap_ratio': 0.3,     # Sequence overlap for better training
        'shuffle_data': True,
        'num_workers': 0,
        
        # Hardware
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'pin_memory': True,
    }

if __name__ == "__main__":
    import torch
    
    # Test data creation
    texts = create_comprehensive_dataset()
    config = get_training_config()
    
    print("Data creation test completed successfully!")
    print(f"Sample text preview: {texts[0][:200]}...")
    print(f"Configuration: {config}")
