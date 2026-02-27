# Chain of Thought Reasoning Module

## Overview

The Chain of Thought (CoT) reasoning module implements step-by-step logical reasoning for the Alan model. This approach breaks down complex problems into manageable steps, leading to more coherent and well-structured responses.

## What is Chain of Thought?

Chain of Thought is a prompting technique that encourages language models to generate intermediate reasoning steps before producing a final answer. Instead of jumping directly to a conclusion, the model works through:

1. **Analysis** - Understanding the problem
2. **Decomposition** - Breaking it into parts
3. **Pattern Recognition** - Identifying relationships
4. **Inference** - Applying logic
5. **Synthesis** - Creating a coherent conclusion

## Implementation Details

### ChainOfThought Class

The `ChainOfThought` class handles all reasoning operations:

```python
from reasoning.chain_of_thought import ChainOfThought
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize
model = GPT2LMHeadModel.from_pretrained("../models/gpt-neo-1.3b")
tokenizer = GPT2Tokenizer.from_pretrained("../models/gpt-neo-1.3b")
cot = ChainOfThought(model, tokenizer, max_reasoning_steps=5)

# Generate with reasoning
reasoning_steps, final_response = cot.generate_with_reasoning("Your prompt here")
```

### Key Methods

#### `generate_reasoning_steps(prompt: str) -> List[str]`
Generates a series of reasoning steps for the given prompt.

#### `generate_with_reasoning(prompt: str, max_length: int) -> Tuple[List[str], str]`
Generates both the reasoning process and final response.

#### `format_reasoning_output(reasoning_steps: List[str]) -> str`
Formats reasoning steps into readable output.

## Features

### 1. **Multi-Step Reasoning**
- Configurable number of reasoning steps (1-7)
- Each step builds on previous reasoning
- Gradual refinement of understanding

### 2. **Intelligent Step Titling**
- Dynamic step descriptions
- Contextual naming based on step position
- Clear semantic progression

### 3. **Response Cleaning**
- Removes incomplete sentences
- Eliminates special tokens
- Normalizes whitespace
- Prevents answer hallucination

### 4. **Context Management**
- Handles token limits gracefully
- Prevents overflow errors
- Maintains coherence across steps

### 5. **Output Formatting**
- Structured, readable output
- Visual separators
- Clear step demarcation

## Generation Parameters

The module uses optimized parameters for quality reasoning:

```python
{
    "num_beams": 2,           # Beam search width
    "temperature": 0.7,       # Creativity level (0-2)
    "top_p": 0.9,             # Nucleus sampling
    "do_sample": True,        # Use sampling instead of greedy
    "early_stopping": True,   # Stop at logical conclusion
}
```

## Usage Examples

### Basic Reasoning

```python
from reasoning.chain_of_thought import ChainOfThought

cot = ChainOfThought(model, tokenizer)
steps, response = cot.generate_with_reasoning(
    "Explain quantum computing in simple terms"
)

print(cot.format_reasoning_output(steps))
print(f"Final Answer: {response}")
```

### Custom Configuration

```python
# Deep reasoning with more steps
cot = ChainOfThought(model, tokenizer, max_reasoning_steps=7)

# Shorter reasoning for quick queries
cot = ChainOfThought(model, tokenizer, max_reasoning_steps=2)
```

### Step-by-Step Generation

```python
# Get just the reasoning without final response
reasoning_steps = cot.generate_reasoning_steps(prompt)

for i, step in enumerate(reasoning_steps, 1):
    print(f"Step {i}:\n{step}\n")
```

## Output Example

```
============================================================
REASONING PROCESS (Chain of Thought)
============================================================

Step 1 - Initial Analysis:
Understanding the core concepts and breaking down...
--------------------------------------------------

Step 2 - Breaking Down the Problem:
Identifying the key components and relationships...
--------------------------------------------------

Step 3 - Identifying Key Patterns:
Recognizing recurring themes and patterns...
--------------------------------------------------

Final Conclusion:
Synthesizing all reasoning into a coherent answer...
============================================================
```

## Performance Optimization

### Token Management
- Truncates long inputs gracefully
- Limits context to prevent memory overflow
- Efficient tokenization

### Inference Speed
- Uses beam search for balance between quality and speed
- Configurable inference parameters
- Early stopping when conclusion is reached

### Memory Efficiency
- Processes with `torch.no_grad()` to save memory
- Cleans intermediate tensors
- Optimized token limits

## Advanced Configuration

### Adjusting Temperature

```python
# Lower temperature = more deterministic
"temperature": 0.5  # More focused reasoning

# Higher temperature = more creative
"temperature": 1.0  # More exploratory reasoning
```

### Beam Search Width

```python
# Single beam = fastest
"num_beams": 1

# Multiple beams = better quality
"num_beams": 3
```

### Custom Step Titles

Extend the `_get_step_title()` method to customize step descriptions:

```python
def _get_step_title(self, step_num: int) -> str:
    custom_titles = {
        1: "Problem Understanding",
        2: "Information Gathering",
        # ... add more
    }
    return custom_titles.get(step_num, f"Step {step_num}")
```

## Reasoning Techniques Implemented

### 1. **Step-by-Step Decomposition**
Breaks complex problems into smaller, manageable parts.

### 2. **Intermediate State Tracking**
Each step builds upon the previous reasoning.

### 3. **Multi-Path Exploration**
Uses beam search to explore multiple reasoning paths.

### 4. **Coherence Checking**
Ensures responses are logically coherent.

### 5. **Context Awareness**
Maintains consistency with previous reasoning steps.

## Testing the Module

Run the test function:

```bash
python -m reasoning.chain_of_thought
```

This will demonstrate:
- Model loading
- Reasoning step generation
- Final response generation
- Output formatting

## Limitations

- Processing time increases with more reasoning steps
- Memory usage grows with context length
- May hallucinate connections between unrelated concepts
- Limited to 1024 token context window

## Future Enhancements

- [ ] Multi-model reasoning ensemble
- [ ] External knowledge integration
- [ ] Confidence scoring for each step
- [ ] Multi-language support
- [ ] Reinforcement learning from feedback
- [ ] Graph-based reasoning visualization

## Integration with Alan Chat

The Chain of Thought module integrates seamlessly with Alan Chat:

```python
# In AlanChat class
self.cot = ChainOfThought(self.model, self.tokenizer, max_reasoning_steps=4)

# When processing messages
reasoning_steps, response = self.cot.generate_with_reasoning(user_input)
```

Users can toggle reasoning on/off during chat with the `reasoning` command.

## Best Practices

1. **Use for Complex Questions**: Enable reasoning for nuanced topics
2. **Disable for Quick Queries**: Use direct generation for simple questions
3. **Monitor Token Usage**: Keep eye on context size
4. **Review Reasoning Steps**: Verify logic before trusting responses
5. **Adjust Parameters**: Fine-tune based on use case

## References

- Original Chain of Thought paper: Wei et al. (2022)
- GPT-2 Architecture: Radford et al. (2019)
- Prompt Engineering: Brown et al. (2020)

---

For integration with the Alan chat system, see [Alan Chat Documentation](../cht/README.md)
