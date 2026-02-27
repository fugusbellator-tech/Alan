#!/usr/bin/env python3
"""Verify the downloaded model can be loaded."""
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

model_dir = Path('/workspaces/Alan/models/gpt-neo-1.3b')

# Check files
files = {f.name: f.stat().st_size / (1024**3) for f in model_dir.glob('*') if f.is_file()}
print("Model files:")
for name, size in sorted(files.items()):
    print(f"  {name}: {size:.2f} GB")

print(f"\nTotal: {sum(files.values()):.2f} GB\n")

# Try loading
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map=None,
    low_cpu_mem_usage=True,
)
print(f"✓ Model loaded: {model.config.model_type}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

# Quick test
print("\nQuick inference test...")
inputs = tokenizer("Hello, how are", return_tensors="pt")
with __import__('torch').no_grad():
    outputs = model.generate(**inputs, max_new_tokens=10)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"✓ Model inference works!")
print(f"  Input: 'Hello, how are'")
print(f"  Output: '{response}'")

print("\n✅ ALL CHECKS PASSED - Model is ready to use!")
