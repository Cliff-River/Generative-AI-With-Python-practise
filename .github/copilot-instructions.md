# Copilot Instructions

# Project Overview
This project is a comprehensive practice following a tutorial series on building generative AI applications using Python. It includes various demos and experiments showcasing pretrained models, prompt engineering, LLM orchestration, and vector databases.

## Project Structure
The project follows a modular course-style structure:
- **c02_pretrained_model/**: Pre-trained transformer models (text generation, translation, NER, QA, image/audio generation, NLI)
- **c03_large_language_model/**: LLM integration demos (OpenAI, Groq, local LLMs, chaining, routing, parallel processing)
- **c04_prompt_engineering/**: Advanced prompt techniques (few-shot learning, self-consistency, feedback loops, chaining)
- **c05_vector_databases/**: RAG and vector database implementations (chunking, embeddings, storage, retrieval)
- **miscellany/**: Utility scripts and experimental features (agents, custom tools, testing)
- **tests/**: Comprehensive test suite following main codebase structure

## Project Snapshot
- Course-style playground for generative AI recipes; each `c0X_` folder is a standalone demo module
- Python 3.11+ targeted via `pyproject.toml`; local `.python-version` pins 3.13 for uv/pyenv consistency
- Uses `uv` for Python virtual environment and dependency management
- Dependencies center on `langchain`, `transformers`, Hugging Face loaders, and multiple hosted LLM providers (OpenAI, Groq, Gemini, Ollama, etc.)
- GPU-accelerated libraries (`torch`, `diffusers`, `tensorflow`) are optional and loaded lazily to avoid heavy downloads

# Development Conventions
- Use `# %%` to split code cells in notebooks for step-by-step execution
- Use `# %% [markdown]` to add explanatory comments or markdown blocks

## Environment & Secrets
- Nearly every script expects `.env` values; common keys include:
  - `OPENAI_API_KEY` for OpenAI GPT models
  - `GROQ_API_KEY` for Groq LLM API
  - Google/Gemini credentials for Google's models
  - Other LLM provider API keys as needed
- Load order uses `load_dotenv(find_dotenv(...))`, so ensure `.env` lives at workspace root
- Network calls hit external services (Gutenberg, Wikipedia, LangChain Hub); mock or guard them when adding automated tests
- Handle missing API keys gracefully with informative logging

## Code Patterns
- Scripts follow VS Code notebook style with cell separators; preserve them for step-by-step execution
- Demos rely on declarative metadata; e.g., `c05_vector_databases/chunking/custom_chunking.py` annotates Gutenberg docs, then pipes through custom splitters (`custom_chunking_utils.py`)
- Vector helpers live under `c05_vector_databases/chunking/` and `c05_vector_databases/data_loder/` (note: directory name contains typo)
- `miscellany/agent_write_md.py` wires a LangChain ReAct agent with a local tool that overwrites `story.md`
- Follow this pattern for new agents/tools: define tool, register in `Tool(...)`, then compose with `AgentExecutor`
- Use lazy imports for heavy ML libraries to maintain fast startup times
- Document function parameters and return types with type hints
- Follow PEP 8 style guidelines and existing code patterns

## Testing & Quality
- The project uses `pytest` to discover `unittest` cases. Run with `uv run pytest` from repo root
- Tests enforce very specific Roman numeral splitting and carriage-return handling (`\r\n`)
- When modifying regexes, mirror those expectations and expand test coverage rather than relaxing assertions
- For new utility modules, place test cases in `tests/` using the same pattern (manual `sys.path` injection)
- Test directory structure mirrors the main codebase for easy navigation
- Mock external API calls in tests to avoid network dependencies
- Maintain high test coverage for data processing and utility functions

## Collaboration Tips
- Favor isolated scripts per experiment; match folder conventions (`c0X_topic/feature.py`)
- Document required environment variables inline and log when external calls are skipped due to missing keys
- Reuse shared resources (e.g., `DEFAULT_EMBEDDINGS` in `embeddings_util.py`) to avoid repeated model downloads
- When adding long-running or I/O heavy operations, gate them behind `if __name__ == "__main__"` so imports stay lightweight
- Use descriptive function and variable names that explain the AI/ML concepts being demonstrated
- Include example usage and expected outputs in docstrings
- Add error handling for model loading and API failures
- Keep model configurations external to code (use config files or environment variables)

## VS Code Setup
- Recommended extensions:
  - Python (Microsoft)
  - Pylance (Microsoft)
  - Jupyter (Microsoft)
  - GitLens (optional)
- Use the Python interpreter from the `uv` virtual environment
- Enable type checking for better development experience
- Use the notebook interface for exploratory work with cell separators

## Development Workflow
1. **Setup Environment**:
   ```bash
   uv sync  # Install dependencies
   # or
   pip install -e .  # If uv unavailable
   ```

2. **Create New Demo**:
   - Add to appropriate `c0X_` folder
   - Include cell separators for cell-based execution
   - Document required environment variables
   - Add minimal test if core functionality is complex

3. **Test Changes**:
   ```bash
   uv run pytest
   python -m compileall  # Quick syntax check
   ```

4. **Common Development Tasks**:
   - Add new LLM provider → Update `c03_large_language_model/`
   - New prompt technique → Add to `c04_prompt_engineering/`
   - Vector/RAG improvement → Extend `c05_vector_databases/`
   - New model type → Add to `c02_pretrained_model/`

## Handy Commands
- **Environment Setup**:
  ```bash
  uv add <package>        # Add dependency
  uv sync                 # Sync all dependencies
  pip install -e .       # Alternative if uv unavailable
  ```

- **Testing & Quality**:
  ```bash
  uv run pytest          # Run all tests
  python -m compileall   # Quick syntax/lint check
  ```

- **Development**:
  ```bash
  # Run specific demo
  python c03_large_language_model/chat_with_openai.py
  
  # Check Python path
  python -c "import sys; print(sys.path)"
  ```

## Model Management
- Models are loaded on-demand to avoid memory overhead
- Use model-specific caching directories to avoid re-downloads
- For large models, provide fallback to smaller alternatives
- Log model loading times and memory usage for transparency
- Consider using model quantization for local LLMs when possible

## Version Control
- **Commit Message Format**: 
  - `feat: add new prompt chaining demo`
  - `fix: resolve chunking boundary issues in custom splitter`
  - `test: add coverage for embeddings utility functions`
  - `docs: update API requirements for new LLM providers`
- Keep commits focused on single concepts or features
- Include issue numbers or feature references when applicable
- Generate meaningful commit messages that summarize changes clearly
- Add co-authors for significant contributions or pair programming sessions

## Performance Considerations
- Use async/await for multiple API calls where possible
- Implement batching for vector operations
- Cache frequently used model outputs
- Monitor memory usage with large models
- Use progress bars for long-running operations
- Consider model size when choosing between local vs. API calls