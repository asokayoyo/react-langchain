# Copilot Instructions for react-langchain

## Project Overview

This is a LangChain-based project implementing the **ReAct (Reasoning + Acting) pattern** for AI agents. The ReAct pattern enables LLMs to reason step-by-step and take actions using tools.

## Architecture

- **Single-file application** (`main.py`) - Contains tools, prompts, and agent logic
- **ReAct Prompt Template** - Uses the classic Thought/Action/Observation cycle format
- **Tool-based execution** - Tools are decorated with `@tool` from `langchain.tools`

## Key Patterns

### Tool Definition
Tools use the LangChain `@tool` decorator with descriptive docstrings (used as tool descriptions by the LLM):

```python
@tool
def my_tool(param: str) -> ReturnType:
    """Description of what this tool does - this becomes the tool's description."""
    # Implementation
```

### ReAct Prompt Structure
The project uses a specific prompt format with these sections in order:
1. `Question:` - The input question
2. `Thought:` - LLM's reasoning
3. `Action:` - Tool name to use
4. `Action Input:` - Input for the tool
5. `Observation:` - Tool result
6. `Final Answer:` - Concluding response

## Development

### Environment Setup
```bash
# Uses uv for dependency management (pyproject.toml)
uv sync

# Requires .env file with API keys (e.g., OPENAI_API_KEY)
```

### Code Style
- **Formatter**: `black` (configured in dependencies)
- **Import sorting**: `isort`
- Python 3.13+ required

### Running
```bash
uv run python main.py
```

## Dependencies

- `langchain` / `langchain-openai` - Core LLM framework
- `python-dotenv` - Environment variable loading from `.env`

## When Adding New Tools

1. Define with `@tool` decorator in `main.py`
2. Include clear docstring (LLM uses this to understand the tool)
3. Add to the `tools` list in `main()`
4. Tools are automatically formatted into the prompt via `{tools}` and `{tool_names}`
