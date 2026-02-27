"""
Alan Chat Interface
A comprehensive chat system with the Alan model (GPT-2) using Chain of Thought reasoning.
All models are loaded locally for optimal performance.
Includes meta-learning abilities and code execution capabilities.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import torch

# Add reasoning module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reasoning'))
# Add tools to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from chain_of_thought import ChainOfThought
from .fast_inference import FastInferenceOptimizer

# Import Alan's tools
try:
    from abilities import AlanAbilities
    from python_executor import PythonExecutor
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False


class AlanChat:
    """
    Chat interface for the Alan model with Chain of Thought reasoning.
    Uses locally loaded GPT-2 models for fast inference.
    """
    
    def __init__(self, model_path: str = "models/gpt-neo-1.3b", use_cot: bool = False, 
                 use_gpu: bool = False):
        """
        Initialize the Alan chat system.
        
        Args:
            model_path: Path to the pre-trained model
            use_cot: Whether to use Chain of Thought reasoning
            use_gpu: Whether to use GPU if available
        """
        # normalize path in case a relative path is provided
        self.model_path = os.path.abspath(model_path)
        self.use_cot = use_cot
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        print("="*60)
        print("Alan - Advanced Language AI with Reasoning")
        print("="*60)
        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device}")
        print(f"Chain of Thought: {'Enabled' if use_cot else 'Disabled'}")
        if TOOLS_AVAILABLE:
            print("Tools: ✓ Enabled (Meta-Learning, Code Execution)")
        print()
        
        # Load model and tokenizer from local path
        # use local_files_only to avoid huggingface trying to treat the path as a repo id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        
        # Apply INT8 quantization for faster inference (optional, skip if it causes issues)
        # Quantization can sometimes fail with large models due to deepcopy issues
        # The smart caching in fast_inference provides similar speedups anyway
        try:
            import torch.quantization
            # Check if already quantized
            if not any(isinstance(m, torch.nn.quantized.Linear) for m in self.model.modules()):
                # Quantization - skip for now due to deepcopy issues with large models
                # self.model = torch.quantization.quantize_dynamic(...)
                pass
        except Exception as e:
            pass
        
        self.model.to(self.device)
        self.model.eval()
        
        # Compile / optimize model for faster inference if possible
        # NOTE: torch.compile can sometimes cause slowdowns, so disable it
        # if hasattr(torch, "compile") and self.device == "cpu":
        #     try:
        #         self.model = torch.compile(self.model, mode="reduce-overhead")
        #         print("Model compiled for faster inference.")
        #     except Exception as e:
        #         pass

        # If a GPU is being used, switch model to half precision for better throughput
        if self.device == "cuda":
            try:
                self.model.half()
                print("Converted model to FP16 for GPU inference.")
            except Exception:
                pass

        # Initialize chain of thought reasoning
        if self.use_cot:
            self.cot = ChainOfThought(self.model, self.tokenizer, max_reasoning_steps=1)
        
        # Initialize fast inference optimizer with smart caching
        self.fast_inference = FastInferenceOptimizer(self.model, self.tokenizer, cache_size=256)
        
        # Initialize Alan's tools if available
        if TOOLS_AVAILABLE:
            self.abilities = AlanAbilities()
            self.executor = PythonExecutor()
            self.tools_enabled = True
            # optional network helper may be added later
            try:
                from tools import network
                self.network = network
            except ImportError:
                self.network = None
        else:
            self.abilities = None
            self.executor = None
            self.tools_enabled = False
            self.network = None
        
        # Chat history
        self.chat_history = []
        self.conversation_context = ""
        
        # Set special tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Alan is ready! Type 'exit' to quit, 'clear' to clear history,")
        print("'reasoning' to toggle chain of thought, 'history' to view chat history,")
        print("'tools' to see tools status, 'exec <code>' to execute code, or 'help' for more.\n")
    
    def _build_context(self) -> str:
        """Build conversation context from history."""
        if not self.chat_history:
            return ""
        
        context = ""
        for entry in self.chat_history[-5:]:  # Keep last 5 interactions
            context += f"User: {entry['user']}\nAlan: {entry['response']}\n\n"
        
        return context
    
    def generate_response(self, user_input: str, use_reasoning: bool = None) -> str:
        """
        Generate a response using the Alan model.
        
        Args:
            user_input: The user's input message
            use_reasoning: Whether to use chain of thought (overrides settings)
            
        Returns:
            Generated response from Alan
        """
        use_reasoning = use_reasoning if use_reasoning is not None else self.use_cot
        
        try:
            # Build prompt with context
            context = self._build_context()
            full_prompt = f"{context}User: {user_input}\nAlan:"
            
            if use_reasoning:
                print("\n[Processing with Chain of Thought...]")
                reasoning_steps, response = self.cot.generate_with_reasoning(
                    user_input, 
                    max_length=80
                )
                
                # Display reasoning
                print(self.cot.format_reasoning_output(reasoning_steps))
                return response
            else:
                # Direct generation without reasoning - ULTRA-fast path with caching
                print("\n[Processing (ultra-fast mode with caching)...]")
                response = self.fast_inference.fast_generate(user_input, max_tokens=40)
                return response
        
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while processing your request."
    
    def chat(self):
        """Main chat loop."""
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() == 'exit':
                    print("\nAlan: Goodbye! It was nice chatting with you.")
                    self.save_chat_history()
                    break
                
                elif user_input.lower() == 'clear':
                    self.chat_history = []
                    self.conversation_context = ""
                    print("Chat history cleared.")
                    continue
                
                elif user_input.lower() == 'reasoning':
                    self.use_cot = not self.use_cot
                    status = "enabled" if self.use_cot else "disabled"
                    print(f"Chain of Thought reasoning {status}.")
                    continue
                
                elif user_input.lower() == 'history':
                    self.display_chat_history()
                    continue
                
                elif user_input.lower() == 'tools':
                    if self.tools_enabled and self.cot:
                        print(self.cot.display_tools_status())
                    else:
                        print("Tools are not available.")
                    continue
                
                elif user_input.lower().startswith('exec '):
                    self.execute_user_code(user_input[5:])
                    continue
                
                elif user_input.lower().startswith('improve '):
                    improvement_desc = user_input[8:]
                    print(f"Alan: I'll improve my reasoning capabilities: {improvement_desc}")
                    continue
                elif user_input.lower().startswith('fetch '):
                    if self.network:
                        url = user_input[6:].strip()
                        try:
                            result = self.network.fetch(url)
                            print(f"Network response:\n{result[:1000]}")
                        except Exception as e:
                            print(f"Error fetching url: {e}")
                    else:
                        print("Network module not available. Try granting internet access first.")
                    continue
                
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"\nAlan: {response}")

                # if autonomy is enabled, allow Alan to try to act on its own
                if self.tools_enabled and self.abilities and getattr(self.abilities, 'autonomy_enabled', False):
                    acted = self.abilities.interpret_and_execute(response)
                    if acted:
                        print("[Alan took an autonomous action based on his response]")

                # Add to history
                self.chat_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "user": user_input,
                    "response": response,
                    "cot_used": self.use_cot
                })
            
            except KeyboardInterrupt:
                print("\n\nAlan: Goodbye!")
                self.save_chat_history()
                break
    
    def display_chat_history(self):
        """Display the chat history."""
        if not self.chat_history:
            print("No chat history yet.")
            return
        
        print("\n" + "="*60)
        print("CHAT HISTORY")
        print("="*60 + "\n")
        
        for i, entry in enumerate(self.chat_history, 1):
            print(f"{i}. {entry['timestamp']}")
            print(f"   You: {entry['user']}")
            print(f"   Alan: {entry['response']}")
            if entry['cot_used']:
                print("   [Used Chain of Thought reasoning]")
            print()
    
    def save_chat_history(self):
        """Save chat history to file."""
        try:
            history_dir = Path("../models")
            history_dir.mkdir(exist_ok=True)
            
            history_file = history_dir / f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(history_file, 'w') as f:
                json.dump(self.chat_history, f, indent=2)
            
            print(f"Chat history saved to {history_file}")
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def display_help(self):
        """Display help information."""
        print("\n" + "="*60)
        print("ALAN CHAT HELP")
        print("="*60)
        print("Commands:")
        print("  exit        - Exit the chat")
        print("  clear       - Clear chat history")
        print("  reasoning   - Toggle Chain of Thought reasoning")
        print("  history     - View chat history")
        print("  tools       - Show tools status (meta-learning, code exec)")
        print("  exec <code> - Execute Python code")
        print("  improve <desc> - Suggest improvement to Alan")
        print("  help        - Display this help message")
        print("\nFeatures:")
        print("  - Chain of Thought reasoning for complex problems")
        print("  - Conversation context awareness")
        print("  - Meta-learning abilities (source code modification)")
        print("  - Python code execution engine")
        print("  - Autonomy mode: Alan can interpret his own responses and execute actions")
        print('  - Network fetch command ("fetch <url>") once internet is granted')
        print("  - Local model inference (no external API calls)")
        print("  - Chat history saving")
        print("="*60 + "\n")
    
    def execute_user_code(self, code: str):
        """Execute Python code provided by user."""
        if not self.tools_enabled:
            print("Code execution is not available.")
            return
        
        print("\n[Analyzing and executing code...]")
        
        # Analyze first
        analysis = self.executor.analyze_code(code)
        
        if not analysis.get("valid"):
            print(f"✗ Syntax Error: {analysis.get('error')}")
            return
        
        if analysis.get("dangerous_calls"):
            print(f"⚠ Warning: Detected potentially dangerous calls: {analysis['dangerous_calls']}")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm != 'y':
                return
        
        # Execute
        success, output, error = self.executor.execute(code)
        
        if success:
            print(f"✓ Execution successful:")
            print(output)
        else:
            print(f"✗ Execution failed:")
            print(str(error))



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Chat with Alan - an intelligent GPT-2 model with reasoning"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../models/gpt2-small",
        help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable Chain of Thought reasoning"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available"
    )
    
    args = parser.parse_args()
    
    # Initialize and start chat
    chat = AlanChat(
        model_path=args.model_path,
        use_cot=not args.no_reasoning,
        use_gpu=args.gpu
    )
    chat.chat()


if __name__ == "__main__":
    main()
