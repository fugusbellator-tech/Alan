# Alan Chat System

## Overview

Alan is a sophisticated chat interface powered by GPT-2 with integrated **Chain of Thought (CoT) reasoning**. The system generates step-by-step logical reasoning before providing responses, ensuring more coherent and well-thought-out answers.

All models are loaded **locally** for optimal performance and privacy.

## Features

- **Chain of Thought Reasoning**: Multi-step logical reasoning process
- **Local Inference**: All models are loaded from local storage (no external API calls)
- **Conversation Context**: Maintains chat history for contextual awareness
- **Interactive Commands**: Rich set of commands to control the chat
- **GPU Support**: Optional GPU acceleration for faster inference
- **Chat History Saving**: Automatic saving of conversations

## Directory Structure

```
Alan/
├── models/
│   └── gpt-neo-1.3b/          # Pre-trained model (locally loaded)
├── reasoning/
│   └── chain_of_thought.py  # Chain of Thought reasoning module
└── cht/
    ├── chat.py              # Main chat interface
    └── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers library

### Setup

```bash
# Install required dependencies
pip install torch transformers

# Navigate to the chat directory
cd cht

# Run the chat
python chat.py
```

## Usage

### Basic Chat

```bash
python chat.py
```

This will start the chat interface. Type your messages and press Enter to chat with Alan.

### Command Line Options

```bash
python chat.py --help
```

Options:
- `--model-path PATH`: Specify custom model path (default: `../models/gpt-neo-1.3b`)
- `--no-reasoning`: Disable Chain of Thought reasoning
- `--gpu`: Use GPU if available

### Example Commands

```bash
# Chat with GPU support
python chat.py --gpu

# Chat without reasoning
python chat.py --no-reasoning

# Use custom model path
python chat.py --model-path /path/to/model
```

## Interactive Commands

While chatting, you can use these special commands:

| Command | Description |
|---------|-------------|
| `exit` | Exit the chat and save history |
| `clear` | Clear chat history |
| `reasoning` | Toggle Chain of Thought reasoning on/off |
| `history` | View chat history |
| `help` | Display help information |

## Chain of Thought Reasoning

When enabled, Alan processes your question through multiple reasoning steps:

1. **Initial Analysis**: Understands the core of your question
2. **Breaking Down**: Deconstructs complex problems
3. **Pattern Recognition**: Identifies key patterns and relationships
4. **Logical Inference**: Applies logic to derive insights
5. **Conclusion**: Synthesizes all reasoning into a final answer

Example output:
```
============================================================
REASONING PROCESS (Chain of Thought)
============================================================

Step 1 - Initial Analysis:
Breaking down the structure of the problem...
...
Final Conclusion:
Based on the reasoning above...
============================================================
```

## Architecture Details

### Chain of Thought Module (`reasoning/chain_of_thought.py`)

The `ChainOfThought` class implements:
- Step-by-step reasoning generation
- Temperature-controlled sampling
- Multi-beam decoding for better quality
- Response cleaning and formatting
- Maximum reasoning depth control

### Chat Interface (`cht/chat.py`)

The `AlanChat` class provides:
- Interactive chat loop
- Context building from conversation history
- Local model management
- Special command handling
- Conversation persistence

## Performance Optimization

The system uses several optimization techniques:

1. **Local Model Loading**: All models are pre-downloaded to avoid network overhead
2. **Device Management**: Automatic CPU/GPU selection
3. **Input Limiting**: Truncates long contexts to prevent memory overflow
4. **Efficient Tokenization**: Uses fast tokenizers for quick preprocessing
5. **Beam Search**: Uses beam search for higher quality outputs

## Model Details

- **Architecture**: GPT-2 Small (117M parameters)
- **Tokenizer**: GPT-2 Tokenizer
- **Context Window**: Up to 1024 tokens
- **Generation Method**: Nucleus sampling with beam search
- **Temperature**: 0.7-0.8 for balanced creativity and coherence

## Example Interaction

```
You: What are the key principles of machine learning?

[Processing with Chain of Thought...]
============================================================
REASONING PROCESS (Chain of Thought)
============================================================
...reasoning steps...
============================================================

Alan: Machine learning is fundamentally based on three 
principles: learning from data, pattern recognition, and 
generalization. The key principles include supervised learning 
for labeled data, unsupervised learning for unlabeled patterns, 
and reinforcement learning for decision-making...
```

## Troubleshooting

### Model not found
Ensure the models are downloaded in `/workspaces/Alan/models/gpt-neo-1.3b/`

### Out of memory
- Reduce `max_reasoning_steps` in the code
- Use `--no-reasoning` flag
- Close other applications

### Slow inference
- Enable `--gpu` if you have a compatible GPU
- Reduce `max_length` parameters in the code

## Advanced Usage

### Custom Reasoning Configuration

Edit the initialization in `chat.py`:
```python
self.cot = ChainOfThought(
    self.model, 
    self.tokenizer, 
    max_reasoning_steps=5  # Increase for deeper reasoning
)
```

### Adjusting Generation Parameters

Modify these in `chain_of_thought.py` and `chat.py`:
- `temperature`: Higher = more creative (0.0-2.0)
- `top_p`: Nucleus sampling cutoff (0.0-1.0)
- `num_beams`: Beam search width (1-3)

## Future Enhancements

- [ ] Fine-tuning capability on custom datasets
- [ ] Multi-agent reasoning
- [ ] Knowledge graph integration
- [ ] Real-time web search integration
- [ ] Voice input/output
- [ ] Web-based interface

## Notes

- The first run may be slower as models are being loaded into memory
- Reasoning takes longer but produces more coherent responses
- Disabling reasoning provides faster responses for quick queries
- Chat history is automatically saved with timestamps

## License

This project uses pre-trained models from Hugging Face.

## Contributing

Contributions are welcome! Feel free to extend the reasoning capabilities or improve the chat interface.
