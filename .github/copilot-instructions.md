# Copilot Instructions

# Project Overview
This project is a practice following a tutorial series on building generative AI applications using Python. It includes various demos and experiments showcasing prompt engineering, LLM orchestration, vector databases, and more.

## Project Snapshot
- Course-style playground for generative AI recipes; each `c0X_` folder is a standalone demo (pretrained models, LLM orchestration, prompt engineering, vector DB prep).
- Python 3.11+ targeted via `pyproject.toml`; local `.python-version` pins 3.13 for uv/pyenv consistency.
- Uses `uv` for Python virtual environment and dependency management.
- Dependencies center on `langchain`, `transformers`, Hugging Face loaders, and multiple hosted LLM providers (OpenAI, Groq, Gemini, Ollama, etc.).

## Environment & Secrets
- Nearly every script expects `.env` values; common keys include `OPENAI_API_KEY`, `GROQ_API_KEY`, Google/Gemini credentials. Load order uses `load_dotenv(find_dotenv(...))`, so ensure `.env` lives at workspace root.
- GPU-accelerated libs (`torch`, `diffusers`, `tensorflow`) are optional; keep imports lazy when possible to avoid unnecessary heavy downloads.
- Network calls hit external services (Gutenberg, Wikipedia, LangChain Hub); mock or guard them when adding automated tests.

## Code Patterns
- Scripts follow VS Code notebook style with `#%%` cell markers; preserve them so users can step-run cells.
- Demos rely on declarative metadata; e.g., `c05_vector_databases/chunking/custom_chunking.py` annotates Gutenberg docs, then pipes through custom splitters (`custom_chunking_utils.py`). Extend by reusing those helpers rather than reimplementing regex logic.
- Vector helpers live under `c05_vector_databases/chunking/` and `.../data_loder/`; note the nonstandard module naming when importing.
- `miscellany/agent_write_md.py` wires a LangChain ReAct agent with a local tool that overwrites `story.md`. Follow this pattern when adding new agents/tools: define tool, register in `Tool(...)`, then compose with `AgentExecutor`.

## Testing & Quality
- The project uses `pytest` to discover `unittest` cases. Run with `uv run pytest` from repo root so `sys.path` insertion stays valid.
- The tests enforce very specific Roman numeral splitting and carriage-return handling (`\r\n`). When modifying regexes, mirror those expectations and expand test coverage rather than relaxing assertions.
- For new utility modules, prefer placing cases in `tests/` using the same pattern (manual `sys.path` injection) until packaging is formalized.

## Collaboration Tips
- Favor isolated scripts per experiment; match folder conventions (`c0X_topic/feature.py`).
- Document required environment variables inline (see `chat_with_openai.py`, `prompt_chaining.py`) and log when external calls are skipped due to missing keys.
- Reuse shared resources (e.g., `DEFAULT_EMBEDDINGS` in `embeddings_util.py`) to avoid repeated model downloads.
- When adding long-running or I/O heavy operations, gate them behind `if __name__ == "__main__"` so imports stay lightweight for other notebooks.

## Handy Commands
- Install deps with `uv add` (preferred) or `uv sync` or `pip install -e .` if uv is unavailable.
- Run tests via `uv run pytest`.
- For quick linting, `python -m compileall` works without extra tooling; no dedicated formatter is configured, so stick to PEP 8 and existing style.
