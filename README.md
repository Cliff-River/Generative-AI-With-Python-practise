# Generative AI with Python - Practice Projects

A comprehensive, course-style playground for building and experimenting with generative AI applications using Python. This project demonstrates practical implementations across the entire spectrum of generative AI: from pre-trained models and large language models to advanced prompt engineering and retrieval-augmented generation (RAG).

## 🚀 Project Structure

### **c02_pretrained_model/** - Pre-trained Transformer Models
Explores ready-to-use transformer models for various NLP and multimodal tasks:
- `fill_mask.py` - Text completion and masking
- `NER.py` - Named Entity Recognition
- `QA.py` - Question Answering systems
- `summarization.py` - Text summarization
- `translation_zh_cn.py` - Chinese language translation
- `zero_shot_classification.py` - Zero-shot text classification
- `text_to_image.py` - Image generation from text
- `text_to_audio.py` - Audio generation from text
- `capstone.py` - Multi-model capstone project

### **c03_large_language_model/** - LLM Integration
Demonstrates integration with multiple LLM providers and orchestration patterns:
- `chat_with_openai.py` - OpenAI GPT models
- `working_with_groq.py`, `groq_gpt.py` - Groq API integration
- `localizing_llm.py` - Local LLM deployment
- `simple_chain.py` - Basic prompt chaining
- `prompt_hub.py` - LangChain Hub integration
- `router_chain.py` - Conditional routing between different models
- `parallel_chain.py` - Parallel processing chains
- `chain_game.py` - Interactive chain-based game
- `llm_stay_on_topic.py` - Topic adherence enforcement
- `image_analyzing.py` - Vision capabilities with LLMs

### **c04_prompt_engineering/** - Advanced Prompt Techniques
Implements sophisticated prompt strategies for improved LLM performance:
- `few-shot.py` - Few-shot learning with examples
- `self_consistency.py` - Self-consistency approach for robust outputs
- `prompt_chaining.py` - Multi-step prompt sequences
- `self_feedback.py` - Self-improvement loops
- `large_rating_small.py` - Rating and feedback mechanisms

### **c05_vector_databases/** - Embeddings & Vector Storage
Comprehensive RAG foundation with embeddings and vector databases:

**chunking/** - Text splitting strategies
- `fixed_size.py` - Fixed-length chunking
- `sematic_chunking.py` - Semantic-aware chunking
- `structed_chunking.py` - Structured document chunking
- `custom_chunking.py` - Custom chunking logic with utilities
- `embeddings_util.py` - Shared embedding models

**data_loder/** - Data ingestion
- `gutenberg_loader.py` - Project Gutenberg books
- `wikipedia_loder.py` - Wikipedia articles
- `txt_file.py` - Local text files
- `load_directory.py` - Batch directory loading

**data_store/** - Vector database operations
- `chroma_database.py` - Chroma vector store setup
- `web_store.py` - Web-based storage integration
- `data_prep.py` - Data preparation pipelines

**embedding/** - Embedding generation
- `word2vec_similaity.py` - Word2Vec similarity metrics
- `sentence_embedding.py` - Sentence-level embeddings
- `embedding_with_langchain.py` - LangChain embedding wrappers

**retrieve_data/** - Vector retrieval
- `chroma_retrieval.py` - Chroma vector retrieval
- `pinecone_retrieval.py` - Pinecone integration

**capstone_project/** - End-to-end RAG application
- `app.py` - Main application
- `data_prep.py` - Data preparation

### **c06_RAG/** - Retrieval-Augmented Generation
Production-ready RAG implementations:
- `simple_RAG.py` - Basic RAG pipeline
- `hybrid_search.py` - Hybrid retrieval (vector + keyword)
- `BM25_TFIDF.py` - BM25 and TF-IDF ranking
- `BM25_TFIDF_zh.py` - Chinese language support

### **miscellany/** - Utilities & Experiments
- `agent_write_md.py` - LangChain ReAct agent with custom tools
- `test_hf_embeddings.py` - Hugging Face embeddings testing

### **tests/** - Test Suite
Mirrors main codebase structure with pytest-based tests:
- Syntax validation and functionality checks
- Vector operation coverage
- Data processing verification
- Mock external API calls to avoid network dependencies

## 🛠️ Setup & Installation

### Prerequisites
- **Python**: 3.11 or higher (3.13 recommended)
- **Package Manager**: `uv` (recommended) or `pip`

### Installation

Using **uv** (recommended):
```bash
# Clone/navigate to project
cd "e:\Documents\Course\Generative AI with Python\practice"

# Sync dependencies
uv sync
```

Using **pip**:
```bash
pip install -e .
```

### Environment Configuration

Create a `.env` file in the project root with required API keys:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Groq
GROQ_API_KEY=gsk_...

# Google Gemini
GOOGLE_API_KEY=...

# Other LLM providers as needed
ANTHROPIC_API_KEY=...
COHERE_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
```

Load order uses `load_dotenv(find_dotenv(...))`, so ensure `.env` is at workspace root.

## 📚 Dependencies

**Core Libraries**:
- `langchain` - LLM orchestration framework
- `transformers` - Hugging Face models
- `python-dotenv` - Environment configuration

**LLM Providers**:
- `openai` - OpenAI GPT models
- `groq` - Groq API client
- `google-generativeai` - Google Gemini

**Vector & Embeddings**:
- `chromadb` - Vector database
- `pinecone-client` - Pinecone integration
- `sentence-transformers` - Embedding models
- `gensim` - Word2Vec models

**ML & Deep Learning** (lazy-loaded):
- `torch` - PyTorch
- `diffusers` - Image generation
- `tensorflow` - TensorFlow models
- `torchaudio` - Audio processing

**Utilities**:
- `pytest` - Testing framework
- `notebook` - Jupyter notebooks
- `jupyterlab` - Jupyter IDE

See `pyproject.toml` for complete dependency list and versions.

## 🚀 Quick Start

### Run a Simple Demo

```bash
# Chat with OpenAI
python c03_large_language_model/chat_with_openai.py

# Vector database example
python c05_vector_databases/chunking/fixed_size.py

# RAG demo
python c06_RAG/simple_RAG.py
```

### Notebook-Style Execution

Code uses `# %%` cell separators for step-by-step execution in VS Code:
- Open any `.py` file in VS Code
- Execute each cell using the cell play button
- Or use `# %% [markdown]` for explanatory comments

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/c05_vector_databases/chunking/test_custom_chunking_utils.py

# Quick syntax check
python -m compileall
```

## 📖 Usage Examples

### Example 1: Text Generation with OpenAI
```python
from c03_large_language_model.chat_with_openai import generate_response

response = generate_response("Explain quantum computing")
print(response)
```

### Example 2: Vector Similarity Search
```python
from c05_vector_databases.chunking.embeddings_util import get_embeddings
from c05_vector_databases.data_store.chroma_database import store_embeddings

# Generate embeddings
embeddings = get_embeddings("Your text here")

# Store and retrieve
store_embeddings(embeddings)
```

### Example 3: RAG Pipeline
```python
from c06_RAG.simple_RAG import RAGPipeline

rag = RAGPipeline()
result = rag.query("Your question here")
```

## 🎯 Key Concepts Covered

| Concept | Module | Examples |
|---------|--------|----------|
| **Pre-trained Models** | c02 | NER, QA, summarization, translation, image/text generation |
| **LLM Integration** | c03 | OpenAI, Groq, local models, multi-provider support |
| **Model Chaining** | c03 | Sequential chains, routing, parallel processing |
| **Prompt Techniques** | c04 | Few-shot, self-consistency, feedback loops, chaining |
| **Vector Embeddings** | c05 | Word2Vec, sentence embeddings, semantic search |
| **Chunking Strategies** | c05 | Fixed-size, semantic, structured document chunking |
| **Retrieval Methods** | c06 | BM25, TF-IDF, hybrid search, RAG |

## 🔧 Development Workflow

### Add New Demo
1. Create file in appropriate `c0X_` folder
2. Include `# %%` cell separators
3. Document required environment variables
4. Add tests if complex functionality

### Commit Message Format
- `feat: add new prompt chaining demo`
- `fix: resolve chunking boundary issues`
- `test: add coverage for embeddings utility`
- `docs: update API requirements`

## ⚙️ Configuration & Performance

### Model Management
- Models load on-demand to avoid memory overhead
- Use model-specific caching directories
- Fallback to smaller models when needed
- Log loading times and memory usage

### Performance Tips
- Use async/await for multiple API calls
- Implement batching for vector operations
- Cache frequently used outputs
- Monitor memory with large models
- Use progress bars for long operations

## 🧪 Testing & Quality

- Uses `pytest` with `unittest` case discovery
- Tests enforce specific formatting expectations
- Mock external API calls to avoid network dependencies
- High coverage for data processing utilities
- Run: `uv run pytest`

## 🎓 Learning Path

**Beginner**:
1. Start with c02 pre-trained models (text generation, classification)
2. Try c03 simple chains and OpenAI integration
3. Explore basic prompting techniques in c04
4. Experiment with notebooks in c02 and c03

**Intermediate**:
1. Build embeddings pipelines in c05
2. Implement vector storage and retrieval
3. Create RAG systems in c06
4. Combine multiple LLMs with routing and parallel chains

**Advanced**:
1. Implement custom chunking and semantic retrieval
2. Build multi-stage RAG with query expansion
3. Create custom tools and agents in miscellany
4. Optimize performance with hybrid search methods

## 📝 VS Code Setup

**Recommended Extensions**:
- Python (Microsoft)
- Pylance (Microsoft)
- Jupyter (Microsoft)
- GitLens (optional)

**Configuration**:
- Use Python from uv virtual environment
- Enable type checking in Pylance
- Use notebook interface for exploration

## 🤝 Contributing

1. Follow existing code patterns and conventions
2. Use descriptive function names explaining AI/ML concepts
3. Include docstrings with examples
4. Add error handling for API failures
5. Keep model configs external (env vars/config files)
6. Maintain or expand test coverage

## 📄 License

This project is for educational purposes.

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Chroma Vector Database](https://docs.trychroma.com/)

## 📞 Support

For issues or questions:
1. Check existing examples in each module
2. Review `.env` configuration
3. Run tests to validate setup
4. Check error logs and API documentation

---

**Last Updated**: December 2025  
**Python Version**: 3.11+ (3.13 recommended)  
**Status**: Active Development  
**Package Manager**: uv
