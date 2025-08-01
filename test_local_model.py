#!/usr/bin/env python3
"""
Quick Test Script for Your Downloaded Vaani Model
Run this script in the same directory as your model.pth and config.json files
"""

import os
import sys
from local_model_loader import VaaniLocalModel

def main():
    print("🚀 Vaani 1.5B Local Model Test")
    print("=" * 40)
    
    # Check for required files
    required_files = ['model.pth', 'config.json']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        print(f"💡 Make sure you have extracted your downloaded ZIP file")
        print(f"   and are running this script in the same directory")
        return
    
    print(f"✅ Found required files: {required_files}")
    
    try:
        # Load the model
        model = VaaniLocalModel('model.pth', 'config.json')
        
        # Interactive testing
        print(f"\n🎮 Interactive Text Generation")
        print(f"Enter prompts to test your model (type 'quit' to exit)")
        print(f"-" * 50)
        
        while True:
            try:
                prompt = input("\n👤 Enter prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                print(f"🤖 Generating...")
                
                # Generate with different settings
                results = []
                
                try:
                    # Conservative generation
                    result1 = model.generate(prompt, max_length=10, temperature=0.7)
                    if result1.strip():
                        results.append(f"Conservative: '{result1}'")
                    
                    # Balanced generation
                    result2 = model.generate(prompt, max_length=15, temperature=1.0)
                    if result2.strip():
                        results.append(f"Balanced: '{result2}'")
                    
                    # Creative generation
                    result3 = model.generate(prompt, max_length=20, temperature=1.3, top_k=100)
                    if result3.strip():
                        results.append(f"Creative: '{result3}'")
                
                except Exception as e:
                    results.append(f"Error: {e}")
                
                if results:
                    for result in results:
                        print(f"   {result}")
                else:
                    print(f"   [No output generated - this is expected with random training data]")
                
                # Show analysis
                print(f"\n🔍 Quick Analysis:")
                try:
                    analysis = model.analyze_prompt(prompt)
                    print(f"   Input: {analysis['token_meanings']}")
                    top_pred = analysis['top_predictions'][0]
                    print(f"   Top prediction: {top_pred['token']} ({top_pred['probability']:.3f})")
                except Exception as e:
                    print(f"   Analysis error: {e}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"   Error: {e}")
        
        print(f"\n👋 Thanks for testing your Vaani 1.5B model!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"💡 Make sure your model files are from the training session")

if __name__ == "__main__":
    main()
