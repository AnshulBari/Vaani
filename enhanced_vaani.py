#!/usr/bin/env python3
"""
Enhanced Vaani LLM - Nemotron-inspired features
Includes streaming responses, conversation history, and advanced sampling
"""

import torch
import pickle
import time
from local_inference import load_model_and_tokenizer

class EnhancedVaani:
    def __init__(self):
        """Initialize Enhanced Vaani with Nemotron-inspired features"""
        print("🚀 Loading Enhanced Vaani...")
        self.model, self.tokenizer, self.device = load_model_and_tokenizer()
        
        if self.model is None:
            raise Exception("Failed to load model")
            
        self.conversation_history = []
        self.system_prompt = "You are Vaani, a helpful and friendly AI assistant."
        
        # Nemotron-inspired parameters
        self.default_params = {
            'temperature': 0.3,    # Lower like Nemotron (0.2)
            'top_p': 0.8,          # Nucleus sampling like Nemotron (0.7)
            'max_tokens': 100,     # Longer context like Nemotron (1024)
            'repetition_penalty': 1.2
        }
        
        print("✅ Enhanced Vaani loaded successfully!")
        print(f"📊 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"🎯 Features: Streaming, Conversation, Advanced Sampling")
    
    def nucleus_sampling(self, logits, top_p=0.8):
        """Implement nucleus (top-p) sampling like Nemotron"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def set_character(self, character_prompt):
        """Set system prompt like Nemotron's roleplay feature"""
        self.system_prompt = character_prompt
        print(f"🎭 Character set: {character_prompt[:50]}...")
    
    def format_conversation(self, user_input):
        """Format conversation like Nemotron API structure"""
        # Keep last 3 exchanges to maintain context
        recent_history = self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
        
        formatted = f"System: {self.system_prompt}\n\n"
        
        for msg in recent_history:
            formatted += f"{msg['role'].title()}: {msg['content']}\n"
        
        formatted += f"User: {user_input}\nAssistant:"
        return formatted
    
    def generate_streaming(self, prompt, **params):
        """Generate text with streaming output like Nemotron API"""
        # Merge default params with user params
        gen_params = {**self.default_params, **params}
        
        word_to_id = self.tokenizer.get('word_to_id', {})
        id_to_word = self.tokenizer.get('id_to_word', {})
        
        # Tokenize input
        words = prompt.lower().split()
        tokens = [word_to_id.get(word, word_to_id.get('<unk>', 1)) for word in words]
        
        if not tokens:
            tokens = [word_to_id.get('<bos>', 2)]
        
        input_ids = torch.tensor(tokens).unsqueeze(0).to(self.device)
        
        generated_words = []
        repetition_count = {}
        
        print("🤖 Vaani: ", end="", flush=True)
        
        with torch.no_grad():
            for step in range(gen_params['max_tokens']):
                try:
                    outputs = self.model(input_ids)
                    logits = outputs[0, -1, :] if outputs.dim() == 3 else outputs[-1, :]
                    
                    # Apply temperature scaling
                    logits = logits / gen_params['temperature']
                    
                    # Apply repetition penalty
                    for token_id, count in repetition_count.items():
                        if count > 2:
                            logits[token_id] = logits[token_id] / gen_params['repetition_penalty']
                    
                    # Apply nucleus sampling (top-p)
                    logits = self.nucleus_sampling(logits, gen_params['top_p'])
                    
                    # Sample next token
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # Update repetition tracking
                    repetition_count[next_token] = repetition_count.get(next_token, 0) + 1
                    
                    # Convert token to word
                    word = id_to_word.get(next_token, '<unk>')
                    
                    # Stop conditions
                    if word in ['<eos>', '<pad>']:
                        break
                    
                    if word == '.' and step > 10 and repetition_count.get(next_token, 0) > 3:
                        print(word, end="", flush=True)
                        break
                    
                    # Stream output (like Nemotron API)
                    if word != '<unk>' or step < 5:  # Allow some <unk> at start
                        print(word + " ", end="", flush=True)
                        generated_words.append(word)
                        time.sleep(0.05)  # Simulate streaming delay
                    
                    # Update input
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(self.device)], dim=1)
                    
                except Exception as e:
                    print(f"\n❌ Generation error: {e}")
                    break
        
        print("\n")  # New line after generation
        return ' '.join(generated_words)
    
    def chat(self, user_input, stream=True, **params):
        """Chat with conversation context like Nemotron"""
        # Format conversation with context
        full_prompt = self.format_conversation(user_input)
        
        # Generate response
        if stream:
            response = self.generate_streaming(full_prompt, **params)
        else:
            from local_inference import generate_text
            response = generate_text(
                self.model, self.tokenizer, full_prompt, 
                max_length=params.get('max_tokens', 100),
                temperature=params.get('temperature', 0.3),
                device=self.device
            )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🧹 Conversation history cleared")
    
    def show_history(self):
        """Display conversation history"""
        print("\n📜 Conversation History:")
        print("=" * 50)
        for i, msg in enumerate(self.conversation_history):
            print(f"{msg['role'].title()}: {msg['content']}")
            if i < len(self.conversation_history) - 1:
                print("-" * 30)
    
    def interactive_chat(self):
        """Interactive chat mode with Nemotron-like features"""
        print("\n💬 Enhanced Vaani Chat Mode")
        print("=" * 50)
        print("🎭 Commands:")
        print("  'set character [prompt]' - Set roleplay character")
        print("  'clear' - Clear conversation history")
        print("  'history' - Show conversation history")
        print("  'params' - Show generation parameters")
        print("  'quit' - Exit chat")
        print("\n🚀 Enhanced features: streaming, context, nucleus sampling")
        print("💡 Try: 'set character You are a helpful coding assistant'")
        
        while True:
            user_input = input("\n🗣️  You: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye! Thanks for chatting with Enhanced Vaani!")
                break
            elif user_input.lower() == 'clear':
                self.clear_history()
                continue
            elif user_input.lower() == 'history':
                self.show_history()
                continue
            elif user_input.lower() == 'params':
                print(f"🎛️  Current parameters: {self.default_params}")
                continue
            elif user_input.lower().startswith('set character '):
                character = user_input[14:]  # Remove 'set character '
                self.set_character(character)
                continue
            elif not user_input:
                continue
            
            # Generate response with streaming
            try:
                self.chat(user_input, stream=True)
            except KeyboardInterrupt:
                print("\n⏸️  Generation stopped by user")
            except Exception as e:
                print(f"\n❌ Error: {e}")

def demo_characters():
    """Demonstrate different character roleplay like Nemotron"""
    vaani = EnhancedVaani()
    
    characters = [
        {
            "name": "Charles the Weatherman",
            "prompt": "Roleplay as Charles the Weatherman, a charismatic and witty meteorologist. Deliver weather information with humor and dramatic flair.",
            "test": "What's the weather like today?"
        },
        {
            "name": "Coding Assistant",
            "prompt": "You are a helpful programming assistant. Explain coding concepts clearly and provide practical examples.",
            "test": "How do I write a for loop?"
        },
        {
            "name": "Friendly Teacher",
            "prompt": "You are a patient and encouraging teacher. Explain complex topics in simple terms with enthusiasm.",
            "test": "What is artificial intelligence?"
        }
    ]
    
    print("🎭 Character Demonstration (Nemotron-style)")
    print("=" * 60)
    
    for char in characters:
        print(f"\n🎪 Testing Character: {char['name']}")
        print(f"📝 Prompt: {char['prompt'][:60]}...")
        print(f"❓ Question: {char['test']}")
        print()
        
        vaani.set_character(char['prompt'])
        vaani.chat(char['test'], stream=True)
        
        input("\n⏳ Press Enter to continue to next character...")
        vaani.clear_history()

def main():
    """Main function with multiple modes"""
    print("🤖 Enhanced Vaani - Nemotron-Inspired Features")
    print("=" * 60)
    print("Choose mode:")
    print("1. Interactive Chat")
    print("2. Character Demo")
    print("3. Quick Test")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        vaani = EnhancedVaani()
        vaani.interactive_chat()
    elif choice == "2":
        demo_characters()
    elif choice == "3":
        vaani = EnhancedVaani()
        print("\n🧪 Quick Test:")
        response = vaani.chat("Tell me about yourself", stream=True)
    else:
        print("Invalid choice. Starting interactive chat...")
        vaani = EnhancedVaani()
        vaani.interactive_chat()

if __name__ == "__main__":
    main()
