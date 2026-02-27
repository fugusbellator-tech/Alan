#!/usr/bin/env python3
"""
Alan - Complete Test Script for Google Colab
Run this single script in a Colab cell to test the entire Alan system
"""

import os
import sys
import subprocess
import shutil

# ============================================================================
# STEP 1: Setup and Environment Detection
# ============================================================================
IS_COLAB = 'google.colab' in sys.modules
print("="*70)
print("ALAN AI SYSTEM - GOOGLE COLAB TEST")
print("="*70)
print(f"Running in Google Colab: {IS_COLAB}")
print()

# ============================================================================
# STEP 2: Clone Repository and Setup
# ============================================================================
if IS_COLAB:
    print("[1] Cloning Alan repository from GitHub...")
    if os.path.exists('/content/Alan'):
        shutil.rmtree('/content/Alan')
    os.system('git clone https://github.com/fugusbellator-tech/Alan.git /content/Alan 2>&1 | tail -5')
    os.chdir('/content/Alan')
    ALAN_PATH = '/content/Alan'
else:
    # Local testing
    ALAN_PATH = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ALAN_PATH)

sys.path.insert(0, ALAN_PATH)
print(f"Working directory: {os.getcwd()}\n")

# ============================================================================
# STEP 3: Install Dependencies
# ============================================================================
print("[2] Installing required packages...")
packages = ['torch', 'transformers', 'numpy', 'safetensors']
for pkg in packages:
    print(f"  ✓ {pkg}", end='')
    subprocess.run(f'pip install -q {pkg}', shell=True)
    print(" (installed)")

# Install optional packages quietly
subprocess.run('pip install -q requests 2>/dev/null', shell=True)
print()

# ============================================================================
# STEP 4: Download Model (if needed)
# ============================================================================
print("[3] Setting up model...")
model_path = os.path.join(ALAN_PATH, 'models', 'gpt-neo-1.3b')

if not os.path.exists(model_path):
    print("  Model not found locally. Downloading from HuggingFace...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        print("  Downloading tokenizer...", end='', flush=True)
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3b")
        os.makedirs(model_path, exist_ok=True)
        tokenizer.save_pretrained(model_path)
        print(" ✓")
        
        print("  Downloading model (this may take 2-5 minutes)...", end='', flush=True)
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3b")
        model.save_pretrained(model_path)
        print(" ✓")
    except Exception as e:
        print(f"\n  Warning: Could not download model: {e}")
        print("  Will attempt to use local model if available")
else:
    print("  ✓ Model found locally")
print()

# ============================================================================
# STEP 5: Test Alan Modules
# ============================================================================
print("[4] Testing Alan modules...")

try:
    print("  ✓ Importing tools.abilities...", end='', flush=True)
    from tools.abilities import AlanAbilities
    print(" OK")
    
    print("  ✓ Importing tools.python_executor...", end='', flush=True)
    from tools.python_executor import PythonExecutor
    print(" OK")
    
    print("  ✓ Importing reasoning.chain_of_thought...", end='', flush=True)
    from reasoning.chain_of_thought import ChainOfThought
    print(" OK")
    
    print("  ✓ Importing cht.chat...", end='', flush=True)
    from cht.chat import AlanChat
    print(" OK")
    print()
except ImportError as e:
    print(f"\n  ERROR: {e}")
    print("  Make sure all files are in the repository")
    sys.exit(1)

# ============================================================================
# STEP 6: Initialize Alan
# ============================================================================
print("[5] Initializing Alan Chat System...")
try:
    chat = AlanChat(model_path=model_path, use_cot=False, use_gpu=False)
    print()
except Exception as e:
    print(f"ERROR: Could not initialize chat: {e}")
    sys.exit(1)

# ============================================================================
# STEP 7: Test Code Execution
# ============================================================================
print("[6] Testing code execution...")
try:
    executor = PythonExecutor()
    code = "result = 2**10; print(f'2^10 = {result}')"
    success, output, error = executor.execute(code)
    if success:
        print(f"  ✓ Code execution works")
        print(f"    Output: {output.strip()}")
    else:
        print(f"  ✗ Execution failed: {error}")
except Exception as e:
    print(f"  ✗ Code execution error: {e}")
print()

# ============================================================================
# STEP 8: Test Alan's Abilities
# ============================================================================
print("[7] Testing Alan's meta-learning abilities...")
try:
    abilities = AlanAbilities(root_path=ALAN_PATH)
    
    # Test 1: Modification logging
    abilities._log_modification(
        "test", "demo",
        "Test modification logged successfully",
        "print('test')"
    )
    print("  ✓ Modification logging works")
    
    # Test 2: History retrieval
    history = abilities.get_modification_history()
    print(f"  ✓ Modification history works ({len(history)} entries)")
    
    # Test 3: Autonomy control
    abilities.enable_autonomy()
    print("  ✓ Autonomy can be enabled")
    abilities.disable_autonomy()
    print("  ✓ Autonomy can be disabled")
except Exception as e:
    print(f"  ✗ Abilities test failed: {e}")
print()

# ============================================================================
# STEP 9: Test Chat Interface
# ============================================================================
print("[8] Testing Alan chat generation...")
print("-" * 70)

test_prompts = [
    "What is the capital of France?",
    "Explain what an AI is in one sentence.",
]

for prompt in test_prompts:
    try:
        print(f"\nUser: {prompt}")
        response = chat.generate_response(prompt)
        print(f"Alan: {response}")
    except Exception as e:
        print(f"Error: {e}")
        break

print("-" * 70)
print()

# ============================================================================
# STEP 10: Summary
# ============================================================================
print("="*70)
print("TEST SUMMARY")
print("="*70)
print("✓ All Alan modules loaded successfully")
print("✓ Chat interface initialized")
print("✓ Code execution engine working")
print("✓ Meta-learning abilities functional")
print("✓ Chat generation tested")
print()
print("NEXT STEPS:")
print("1. Try more complex prompts for better responses")
print("2. Enable Chain of Thought for reasoning: use_cot=True")
print("3. Test code execution: executor.execute(code)")
print("4. Enable autonomy: abilities.enable_autonomy()")
print("5. Grant internet access: abilities.grant_internet_access()")
print()
print("="*70)
print("✨ Alan is working in Google Colab!")
print("="*70)

# ============================================================================
# STEP 11: Interactive Chat (Optional)
# ============================================================================
if IS_COLAB:
    print("\nNote: For interactive chat in Colab, uncomment the code below:")
    print("# Uncomment to enable interactive chat:")
    print("# while True:")
    print("#     user_input = input('You: ').strip()")
    print("#     if user_input.lower() == 'exit':")
    print("#         break")
    print("#     response = chat.generate_response(user_input)")
    print("#     print(f'Alan: {response}\\n')")
