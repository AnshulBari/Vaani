import torch
import pickle

# Load and test your model
def test_vaani_model():
    print("Loading Vaani model...")
    
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('final_model.pth', map_location=device)
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"💻 Device: {device}")
    print(f"📝 Tokenizer type: {type(tokenizer)}")
    
    return model, tokenizer, device

# Run test
model, tokenizer, device = test_vaani_model()