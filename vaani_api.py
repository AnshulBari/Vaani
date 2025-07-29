#!/usr/bin/env python3
"""
Vaani API - Nemotron-style API interface for Vaani LLM
Mimics the OpenAI/Nemotron API structure for easy integration
"""

import torch
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator
from enhanced_vaani import EnhancedVaani

@dataclass
class ChatMessage:
    """Chat message structure like Nemotron API"""
    role: str  # "system", "user", "assistant"
    content: str

@dataclass
class ChatChoice:
    """Chat completion choice like Nemotron API"""
    index: int
    message: ChatMessage
    finish_reason: str

@dataclass
class ChatCompletion:
    """Chat completion response like Nemotron API"""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]

@dataclass
class ChatCompletionChunk:
    """Streaming chunk like Nemotron API"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]

class VaaniAPI:
    """Nemotron-style API for Vaani LLM"""
    
    def __init__(self):
        """Initialize Vaani API"""
        print("🚀 Initializing Vaani API (Nemotron-style)...")
        self.vaani = EnhancedVaani()
        self.model_name = "vaani/vaani-217m-chat"
        print("✅ Vaani API ready!")
    
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "vaani/vaani-217m-chat",
        temperature: float = 0.3,
        top_p: float = 0.8,
        max_tokens: int = 100,
        stream: bool = False
    ):
        """
        Create chat completion like Nemotron API
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (ignored, uses Vaani)
            temperature: Sampling temperature (0.1-2.0)
            top_p: Nucleus sampling parameter (0.1-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
        """
        
        # Extract system prompt and conversation
        system_prompt = "You are Vaani, a helpful AI assistant."
        conversation = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                conversation.append(msg)
        
        # Set system prompt
        self.vaani.set_character(system_prompt)
        
        # Get the last user message
        if conversation and conversation[-1]["role"] == "user":
            user_input = conversation[-1]["content"]
        else:
            user_input = "Hello"
        
        # Generate parameters
        gen_params = {
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens
        }
        
        if stream:
            return self._create_streaming_response(user_input, gen_params)
        else:
            return self._create_complete_response(user_input, gen_params)
    
    def _create_complete_response(self, user_input: str, params: Dict):
        """Create complete (non-streaming) response"""
        # Generate response
        response_content = self.vaani.chat(user_input, stream=False, **params)
        
        # Create response structure like Nemotron
        completion = ChatCompletion(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=self.model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_content
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        return completion
    
    def _create_streaming_response(self, user_input: str, params: Dict) -> Iterator[ChatCompletionChunk]:
        """Create streaming response like Nemotron API"""
        # Format conversation
        full_prompt = self.vaani.format_conversation(user_input)
        
        word_to_id = self.vaani.tokenizer.get('word_to_id', {})
        id_to_word = self.vaani.tokenizer.get('id_to_word', {})
        
        # Tokenize
        words = full_prompt.lower().split()
        tokens = [word_to_id.get(word, word_to_id.get('<unk>', 1)) for word in words]
        input_ids = torch.tensor(tokens).unsqueeze(0).to(self.vaani.device)
        
        chunk_id = f"chatcmpl-{int(time.time())}"
        
        with torch.no_grad():
            for step in range(params['max_tokens']):
                try:
                    outputs = self.vaani.model(input_ids)
                    logits = outputs[0, -1, :] if outputs.dim() == 3 else outputs[-1, :]
                    
                    # Apply temperature and nucleus sampling
                    logits = logits / params['temperature']
                    logits = self.vaani.nucleus_sampling(logits, params['top_p'])
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    word = id_to_word.get(next_token, '<unk>')
                    
                    if word in ['<eos>', '<pad>']:
                        # Send final chunk
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=self.model_name,
                            choices=[{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        )
                        break
                    
                    if word != '<unk>' or step < 3:
                        # Send content chunk
                        yield ChatCompletionChunk(
                            id=chunk_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=self.model_name,
                            choices=[{
                                "index": 0,
                                "delta": {"content": word + " "},
                                "finish_reason": None
                            }]
                        )
                    
                    input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(self.vaani.device)], dim=1)
                    
                except Exception as e:
                    break

def demo_nemotron_api():
    """Demo Vaani API in Nemotron-style"""
    print("🤖 Vaani API Demo (Nemotron-style)")
    print("=" * 50)
    
    # Initialize API
    api = VaaniAPI()
    
    # Example 1: Charles the Weatherman (like Nemotron example)
    print("\n🎭 Example 1: Character Roleplay")
    messages = [
        {
            "role": "system",
            "content": "Roleplay as Charles the Weatherman, a charismatic and witty meteorologist with a knack for entertaining and informing."
        },
        {
            "role": "user", 
            "content": "What's the weather looking like today?"
        }
    ]
    
    print("📤 Request:")
    print(json.dumps(messages, indent=2))
    print("\n📥 Response:")
    
    # Non-streaming response
    completion = api.create_chat_completion(
        messages=messages,
        temperature=0.3,
        top_p=0.8,
        max_tokens=50
    )
    
    print(f"Assistant: {completion.choices[0].message.content}")
    
    # Example 2: Streaming response
    print("\n\n🌊 Example 2: Streaming Response")
    print("📤 Request: Same as above, but streaming=True")
    print("📥 Streaming Response:")
    print("Assistant: ", end="", flush=True)
    
    for chunk in api.create_chat_completion(
        messages=messages,
        temperature=0.3,
        top_p=0.8,
        max_tokens=50,
        stream=True
    ):
        if chunk.choices[0]["delta"].get("content"):
            print(chunk.choices[0]["delta"]["content"], end="", flush=True)
            time.sleep(0.1)  # Simulate streaming delay
    
    print("\n\n✅ API Demo Complete!")

def nemotron_style_usage():
    """Show exact Nemotron-style usage"""
    print("\n📋 Nemotron-Style Usage Example:")
    print("=" * 40)
    
    code_example = '''
# Usage exactly like Nemotron API:
from vaani_api import VaaniAPI

# Initialize
api = VaaniAPI()

# Chat completion (like Nemotron)
completion = api.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain AI in simple terms"}
    ],
    temperature=0.3,
    top_p=0.8,
    max_tokens=100,
    stream=False
)

print(completion.choices[0].message.content)

# Streaming (like Nemotron)
for chunk in api.create_chat_completion(
    messages=[...],
    stream=True
):
    if chunk.choices[0]["delta"].get("content"):
        print(chunk.choices[0]["delta"]["content"], end="")
'''
    
    print(code_example)

if __name__ == "__main__":
    demo_nemotron_api()
    nemotron_style_usage()
