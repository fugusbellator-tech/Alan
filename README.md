# Alan - Advanced Language AI with Reasoning

A comprehensive chat system powered by GPT-2 with advanced **Chain of Thought (CoT) reasoning** capabilities. Alan generates step-by-step logical reasoning before providing responses, ensuring coherent and well-thought-out answers.

## Overview

```
Alan/
├── models/
│   └── gpt-neo-1.3b/              # Pre-trained model (locally loaded)
├── reasoning/
│   ├── chain_of_thought.py      # Chain of Thought reasoning engine
│   └── README.md
├── cht/
│   ├── chat.py                  # Main chat interface
│   ├── launcher.py              # Easy launcher script
│   └── README.md
└── README.md                    # This file
```

## Key Features

- **Chain of Thought Reasoning**: Multi-step logical reasoning before responses
- **Local Inference**: All models pre-downloaded for instant use
- **GPU Support**: Optional GPU acceleration
- **Interactive Chat**: Rich command interface
- **Context Awareness**: Maintains conversation history

## Quick Start

```bash
cd cht
pip install -r requirements.txt
python chat.py
```

## Interactive Commands

- `exit` - Exit and save chat history
- `clear` - Clear conversation history
- `reasoning` - Toggle Chain of Thought on/off
- `history` - View past conversations
- `help` - Display help

## Documentation

- [Chat System Guide](cht/README.md)
- [Reasoning Module Guide](reasoning/README.md)

## Getting Started

Ready to chat? Navigate to `cht/` and run `python chat.py`
