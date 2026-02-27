#!/usr/bin/env python3
"""
Alan Chat Launcher
Provides an easy way to start chatting with Alan
"""

import subprocess
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Launch Alan Chat with various configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py                 # Start with default settings
  python launcher.py -g              # Start with GPU
  python launcher.py --no-reasoning  # Start without reasoning
  python launcher.py -g -m           # GPU + with reasoning (default)
        """
    )
    
    parser.add_argument(
        "-g", "--gpu",
        action="store_true",
        help="Use GPU if available"
    )
    
    parser.add_argument(
        "-m", "--model-path",
        type=str,
        default="../models/gpt-neo-1.3b",
        help="Path to model (default: ../models/gpt-neo-1.3b)"
    )
    
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable Chain of Thought reasoning"
    )
    
    parser.add_argument(
        "-q", "--quick",
        action="store_true",
        help="Quick start with no reasoning (fastest)"
    )
    
    parser.add_argument(
        "-s", "--smart",
        action="store_true",
        help="Smart mode with reasoning (default)"
    )
    
    args = parser.parse_args()
    
    # Build command
    cmd = [sys.executable, "chat.py"]
    
    if args.quick:
        cmd.append("--no-reasoning")
    
    if args.gpu:
        cmd.append("--gpu")
    
    if args.no_reasoning and not args.quick:
        cmd.append("--no-reasoning")
    
    if args.model_path != "../models/gpt-neo-1.3b":
        cmd.extend(["--model-path", args.model_path])
    
    print("="*60)
    print("Starting Alan Chat System")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print("="*60 + "\n")
    
    try:
        subprocess.run(cmd, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n\nAlan Chat closed.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
