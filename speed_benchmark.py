#!/usr/bin/env python3
"""Benchmark the actual inference speed with different token counts."""

import sys
import os
import time
import torch
sys.path.insert(0, '/workspaces/Alan')

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "models/gpt-neo-1.3b"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

print("Applying quantization...")
try:
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    print("✓ Quantized to INT8")
except Exception as e:
    print(f"Warning: {e}")

model.eval()

prompt = "Question: How fast is this inference?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")

print(f"\nBenchmarking with prompt: '{prompt}'")
print(f"Input tokens: {inputs.input_ids.shape[1]}")

# Test different token counts
for max_tokens in [1, 3, 5, 10, 20]:
    print(f"\n--- Generating {max_tokens} tokens ---")
    start = time.time()
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=1.5,
            top_k=20,
            do_sample=True,
            use_cache=True,
            early_stopping=True,
        )
    
    elapsed = time.time() - start
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    
    print(f"Time: {elapsed:.2f}s ({elapsed/max_tokens:.2f}s per token)")
    print(f"Response: {response[:50]}...")
