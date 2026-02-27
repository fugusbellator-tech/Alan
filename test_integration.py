#!/usr/bin/env python3
"""
Final integration test for Alan with EleutherAI/gpt-neo-1.3B model.
Tests that the model is properly loaded and can be used in the chat interface.
"""
from pathlib import Path
import sys

print("\n" + "="*70)
print("ALAN AI SYSTEM - MODEL INTEGRATION VERIFICATION")
print("="*70)

# 1. Verify model files
print("\n[1/4] CHECKING MODEL FILES")
print("-" * 70)

model_dir = Path('/workspaces/Alan/models/gpt-neo-1.3b')
required_files = ['model.safetensors', 'config.json', 'tokenizer.json']

for req_file in required_files:
    file_path = model_dir / req_file
    exists = file_path.exists()
    status = "✓ PRESENT" if exists else "✗ MISSING"
    print(f"  {req_file:.<40} {status}")
    if exists and req_file.endswith('.safetensors'):
        size_gb = file_path.stat().st_size / (1024**3)
        print(f"    └─ Size: {size_gb:.2f} GB")

# 2. Check code integration
print("\n[2/4] CHECKING CODE INTEGRATION")
print("-" * 70)

test_files = {
    'cht/chat.py': 'gpt-neo-1.3b',
    'cht/launcher.py': 'gpt-neo-1.3b',
    'tools/abilities.py': 'gpt-neo-1.3b',
    'reasoning/chain_of_thought.py': 'gpt-neo-1.3b',
}

all_updated = True
for filepath, required_text in test_files.items():
    full_path = Path(f'/workspaces/Alan/{filepath}')
    if full_path.exists():
        content = full_path.read_text()
        has_reference = required_text in content
        status = "✓" if has_reference else "✗"
        print(f"  {filepath:.<40} {status}")
        if not has_reference:
            all_updated = False
    else:
        print(f"  {filepath:.<40} ✗ FILE NOT FOUND")
        all_updated = False

# 3. Test model loading
print("\n[3/4] TESTING MODEL LOADING")
print("-" * 70)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"  Loading tokenizer from {model_dir}...", end=' ')
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("✓")
    
    print(f"  Loading model (~5GB, may take 30-60 seconds)...", end=' ')
    sys.stdout.flush()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map=None,
        low_cpu_mem_usage=False,  # Slightly faster for inference
    )
    print("✓")
    
    model_info = f"{model.config.model_type} ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)"
    print(f"  Model loaded: {model_info}")
    
except Exception as e:
    print(f"✗ ERROR: {e}")
    sys.exit(1)

# 4. Quick inference test
print("\n[4/4] TESTING INFERENCE")
print("-" * 70)

try:
    print("  Input: 'Hello, tell me about AI'")
    inputs = tokenizer("Hello, tell me about AI", return_tensors="pt")
    
    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.9,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Show first 50 chars of output
    preview = response[len("Hello, tell me about AI"):].strip()[:50]
    print(f"  Output: ...{preview}...")
    print(f"  ✓ Inference successful")
    
except Exception as e:
    print(f"  ✗ ERROR: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
if all_updated:
    print("✅ ALL CHECKS PASSED")
    print("\nSUMMARY:")
    print(f"  • Model type: EleutherAI/gpt-neo-1.3B")
    print(f"  • Location: {model_dir}")
    print(f"  • Integration: Complete (all code files updated)")
    print(f"  • Status: Ready for use in Alan")
    print("\nYou can now start the Alan chat with:")
    print(f"  python3 /workspaces/Alan/cht/launcher.py")
else:
    print("⚠️ SOME CHECKS FAILED")
    print("Please review the status above.")
    sys.exit(1)

print("="*70 + "\n")
