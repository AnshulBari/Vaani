import torch
import pickle

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try loading the model - it might be state_dict or full model
    try:
        model_data = torch.load('final_model.pth', map_location=device)
        
        if isinstance(model_data, dict):
            # It's a state_dict, we need to reconstruct the model
            print("Model file contains state_dict, need to reconstruct model...")
            
            # Import your model class (you'll need to adjust this import)
            try:
                from model import SmallLLM  # Adjust import based on your model file
                
                # Get vocab size from the saved model weights
                vocab_size = model_data['token_embedding.weight'].shape[0]
                d_model = model_data['token_embedding.weight'].shape[1]
                
                print(f"Detected vocab_size: {vocab_size}")
                print(f"Detected d_model: {d_model}")
                
                # Create model with same config as training
                model = SmallLLM(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_layers=16, 
                    n_heads=16,
                    d_ff=4096,
                    max_seq_len=512,
                    dropout=0.1
                )
                
                # Load the state dict
                model.load_state_dict(model_data)
                
            except ImportError:
                print("Could not import model class. Please ensure model.py is available.")
                return None, None, None
                
        else:
            # It's a full model object
            model = model_data
            
        model.eval()
        model.to(device)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None
    
    return model, tokenizer, device

def generate_text(model, tokenizer, prompt, max_length=50, device='cpu'):
    """Generate text using the trained model"""
    model.eval()
    
    # Handle dictionary tokenizer format
    word_to_id = tokenizer.get('word_to_id', {})
    id_to_word = tokenizer.get('id_to_word', {})
    
    # Tokenize input
    words = prompt.lower().split()
    tokens = [word_to_id.get(word, word_to_id.get('<unk>', 1)) for word in words]
    
    if not tokens:  # If tokenization failed
        tokens = [word_to_id.get('<bos>', 2)]  # Start with beginning of sequence token
    
    print(f"Tokenized '{prompt}' to: {tokens}")
    
    # Convert to tensor
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            try:
                outputs = model(input_ids)
                
                # Get next token
                if outputs.dim() == 3:
                    logits = outputs[0, -1, :]
                elif outputs.dim() == 2:
                    logits = outputs[-1, :]
                
                next_token = torch.argmax(logits).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
                
                # Stop if we hit end token
                if next_token == word_to_id.get('<eos>', 3):
                    break
                    
            except Exception as e:
                print(f"Error: {e}")
                break
    
    # Decode result
    generated_tokens = input_ids[0].cpu().tolist()
    
    # Convert tokens back to words
    words = []
    for token in generated_tokens:
        word = id_to_word.get(token, '<unk>')
        words.append(word)
    
    # Join words and clean up
    result = ' '.join(words)
    result = result.replace(' <pad>', '').replace('<pad>', '')
    result = result.replace(' <bos>', '').replace('<bos>', '')
    result = result.replace(' <eos>', '').replace('<eos>', '')
    
    return result.strip()

if __name__ == "__main__":
    # Load model
    model, tokenizer, device = load_model_and_tokenizer()
    
    if model is None:
        print("Failed to load model. Exiting...")
        exit(1)
        
    print(f"Model loaded on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Debug tokenizer
    print(f"\nTokenizer info:")
    print(f"Type: {type(tokenizer)}")
    if isinstance(tokenizer, dict):
        print(f"Keys: {list(tokenizer.keys())}")
        print(f"Vocab size: {tokenizer.get('vocab_size', 'Unknown')}")
        word_to_id = tokenizer.get('word_to_id', {})
        print(f"Actual vocab size: {len(word_to_id)}")
        print(f"Sample words: {list(word_to_id.keys())[:10]}")
    else:
        print(f"Attributes: {dir(tokenizer)}")
        if hasattr(tokenizer, 'word2idx'):
            print(f"Vocab size: {len(tokenizer.word2idx)}")
            print(f"Sample words: {list(tokenizer.word2idx.keys())[:10]}")
    
    # Interactive chat
    while True:
        prompt = input("\nEnter prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
            
        result = generate_text(model, tokenizer, prompt, device=device)
        print(f"Generated: {result}")