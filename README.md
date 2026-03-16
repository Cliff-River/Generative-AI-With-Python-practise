# Generative AI with Python (Practice)

Hands-on, course-style experiments for building generative AI applications with Python.

This repository focuses on practical demos across:
- pre-trained models,
- hosted and local LLM integration,
- prompt engineering patterns,
- embeddings/vector databases,
- retrieval-augmented generation (RAG), and
- agentic workflows.

## Project Layout

- `c02_pretrained_model/`: transformer-based tasks (NER, QA, summarization, translation, text-to-image/audio, etc.)
- `c03_large_language_model/`: LLM provider integration and LangChain patterns (routing, parallel chains, safety/topic control)
- `c04_prompt_engineering/`: few-shot, self-consistency, prompt chaining, self-feedback
- `c05_vector_databases/`: chunking, embeddings, loaders, storage, retrieval, capstone app
- `c06_RAG/`: BM25/TF-IDF, hybrid search, query expansion, RAG evaluation
- `c07_agentic_system/`: agentic RAG experiments
- `miscellany/`: utilities and standalone experiments
- `tests/`: pytest suite mirroring core modules

> Note: some folder/file names intentionally contain typos from course progression (for example `data_loder/`, `sematic_chunking.py`, `structed_chunking.py`).

## Requirements

- Python `>=3.11`
- `uv` recommended for dependency/environment management

## Setup

### 1) Install dependencies

```bash
uv sync
```

Alternative:

```bash
pip install -e .
```

### 2) Configure environment variables

Create `.env` at repository root. Common keys used by scripts include:

```env
OPENAI_API_KEY=...
GROQ_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
PINECONE_API_KEY=...
```

Many scripts call `load_dotenv(find_dotenv(usecwd=True))`, so keeping `.env` at the root is important.

## Run Demos

From the repository root:

```bash
uv run python c03_large_language_model/chat_with_openai.py
uv run python c05_vector_databases/chunking/fixed_size.py
uv run python c06_RAG/simple_RAG.py
```

Most demo scripts are notebook-style Python files with `# %%` cells and are also suitable for step-by-step execution in VS Code.

## Testing

Run full test suite:

```bash
uv run pytest
```

Run a specific area:

```bash
uv run pytest tests/c05_vector_databases/chunking
```

Optional syntax check:

```bash
uv run python -m compileall .
```

## Development Notes

- Keep heavy model imports lazy when possible.
- Add new examples to the matching `c0X_` module.
- Preserve `# %%` / `# %% [markdown]` cell structure for tutorial-style execution.
- Add or update tests in `tests/` when changing reusable utilities.

## Dependencies

Core dependencies are managed in `pyproject.toml` and include:
- LangChain ecosystem (`langchain-*`)
- model/tooling (`transformers`, `torch`, `tensorflow`, `diffusers`)
- retrieval/vector stack (`chromadb`, `pinecone`, `sentence-transformers`, `rank-bm25`)
- evaluation/utilities (`ragas`, `pytest`, `datasets`, `wikipedia`)

## License

Educational/practice repository.

---

Last updated: March 2026
