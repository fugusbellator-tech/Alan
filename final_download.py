#!/usr/bin/env python3
"""
Final attempt: Download GPT-Neo 1.3B with minimal dependencies.
Uses huggingface_hub.snapshot_download directly.
"""
import os
import sys
from pathlib import Path

# Ensure clean state
os.chdir('/workspaces/Alan')

SAVE_DIR = Path('/workspaces/Alan/models/gpt-neo-1.3b')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DOWNLOADING GPT-Neo 1.3B")
print("=" * 70)

# Step 1: Use snapshot_download with proper cleanup
from huggingface_hub import snapshot_download

try:
    print("\nDownloading model (real-time mode)...")
    local_path = snapshot_download(
        repo_id='EleutherAI/gpt-neo-1.3B',
        local_dir=str(SAVE_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"\n✓ Download finished: {local_path}")
    
except KeyboardInterrupt:
    print("\nInterrupted.")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Verify what we have
print("\n" + "=" * 70)
print("CHECKING DOWNLOADED FILES")
print("=" * 70)

total_size = 0
file_count = 0
for root, dirs, files in os.walk(SAVE_DIR):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        fsize_mb = os.path.getsize(fpath) / (1024 ** 2)
        fsize_gb = fsize_mb / 1024
        if fsize_gb > 0.01:
            print(f"  {fname}: {fsize_gb:6.2f} GB")
        else:
            print(f"  {fname}: {fsize_mb:6.1f} MB")
        total_size += fsize_mb
        file_count += 1

print(f"\nTotal: {file_count} files, {total_size/1024:.2f} GB")
print("=" * 70)

# Step 3: Quick load test
print("\nTesting model load...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Just test loading - don't keep in memory
    print("  Loading tokenizer...", end=" ", flush=True)
    tok = AutoTokenizer.from_pretrained(str(SAVE_DIR))
    print("✓")
    
    print("  Loading model...", end=" ", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(SAVE_DIR), low_cpu_mem_usage=True)
    print("✓")
    
    print("\n✓ Model loads successfully!")
    
except Exception as e:
    print(f"\n✗ Load test failed: {e}")
    import traceback
    traceback.print_exc()
