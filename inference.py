import torch
import json
import os
from model import SmallLLM
from tokenizer import SimpleTokenizer

class LLMGenerator:
    def __init__(self, model_path, tokenizer_path, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        self.model = SmallLLM(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            n_layers=self.config['n_layers'],
            n_heads=self.config['n_heads'],
            d_ff=self.config['d_ff'],
            max_seq_len=self.config['max_seq_len'],
            dropout=0.0  # No dropout for inference
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        
        print(f"Model loaded on device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
    def generate(self, prompt, max_length=100, temperature=0.8, top_k=50):
        """Generate text given a prompt"""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        generated_tokens = tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits = self.model(input_ids)
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to sequence
                generated_tokens.append(next_token.item())
                
                # Stop if EOS token
                if next_token.item() == self.tokenizer.special_tokens['<eos>']:
                    break
                
                # Update input_ids for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                # Truncate if sequence gets too long
                if input_ids.shape[1] > self.config['max_seq_len']:
                    input_ids = input_ids[:, 1:]
        
        return self.tokenizer.decode(generated_tokens)

def main():
    try:
        # Initialize generator
        generator = LLMGenerator(
            model_path='final_model.pth',
            tokenizer_path='tokenizer.pkl',
            config_path='model_config.json'
        )
        
        # Interactive chat
        print("\nSmall LLM Chat (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            prompt = input("\nYou: ")
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if prompt.strip() == "":
                continue
                
            try:
                print("AI: ", end="", flush=True)
                response = generator.generate(prompt, max_length=50, temperature=0.8)
                print(response)
            except Exception as e:
                print(f"Error generating response: {e}")
                
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please run training first with: python train.py")
        print("Available model files:")
        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if model_files:
            for file in model_files:
                print(f"  - {file}")
        else:
            print("  No model files found")
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    main()
