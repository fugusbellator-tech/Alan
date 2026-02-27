import os
from pathlib import Path

result_file = Path('/workspaces/Alan/download_status.txt')

with open(result_file, 'w') as f:
    model_dir = Path('/workspaces/Alan/models/gpt-neo-1.3b')
    f.write(f"Model directory exists: {model_dir.exists()}\n")
    
    if model_dir.exists():
        files = sorted(model_dir.glob('*'))
        f.write(f"Number of files: {len(files)}\n")
        
        for file in files:
            if file.is_file():
                size_mb = file.stat().st_size / (1024**2)
                f.write(f"{file.name}: {size_mb:.1f} MB\n")
        
        total = sum(f.stat().st_size for f in files if f.is_file()) / (1024**3)
        f.write(f"\nTotal size: {total:.2f} GB\n")
        
        has_weights = any('model' in f.name and f.suffix in ['.safetensors', '.bin'] for f in files)
        f.write(f"Has model weights: {has_weights}\n")

print("Status written to /workspaces/Alan/download_status.txt")
