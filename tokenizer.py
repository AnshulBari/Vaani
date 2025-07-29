import re
import json
from collections import defaultdict, Counter
import pickle

class SimpleTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        
    def train(self, texts):
        """Train tokenizer on text corpus"""
        print("Training tokenizer...")
        
        # Basic word tokenization with subword units
        word_counts = Counter()
        
        for text in texts:
            # Simple preprocessing
            text = text.lower()
            words = re.findall(r'\b\w+\b|[^\w\s]', text)
            word_counts.update(words)
        
        # Create vocabulary
        vocab = list(self.special_tokens.keys())
        
        # Add most frequent words
        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        vocab.extend([word for word, _ in most_common])
        
        # Create mappings
        self.word_to_id = {word: i for i, word in enumerate(vocab)}
        self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        
        print(f"Vocabulary size: {len(self.word_to_id)}")
        
    def encode(self, text):
        """Convert text to token ids"""
        text = text.lower()
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        token_ids = [self.special_tokens['<bos>']]
        for word in words:
            token_id = self.word_to_id.get(word, self.special_tokens['<unk>'])
            token_ids.append(token_id)
        token_ids.append(self.special_tokens['<eos>'])
        
        return token_ids
    
    def decode(self, token_ids):
        """Convert token ids back to text"""
        words = []
        for token_id in token_ids:
            if token_id in [self.special_tokens['<bos>'], self.special_tokens['<eos>']]:
                continue
            word = self.id_to_word.get(token_id, '<unk>')
            words.append(word)
        return ' '.join(words)
    
    def save(self, path):
        """Save tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens
            }, f)
    
    def load(self, path):
        """Load tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word_to_id = data['word_to_id']
            self.id_to_word = data['id_to_word']
            self.vocab_size = data['vocab_size']
            self.special_tokens = data['special_tokens']
