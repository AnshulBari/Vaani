import torch
import pickle

def chat_with_vaani():
    # Load model and tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('final_model.pth', map_location=device)
    model.eval()
    
    print("🤖 Vaani AI is ready! (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        # Simple response (you'll need to implement proper generation)
        print(f"Vaani: Processing '{user_input}'...")
        # Add your generation logic here

if __name__ == "__main__":
    chat_with_vaani()