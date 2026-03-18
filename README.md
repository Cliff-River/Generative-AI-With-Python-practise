# Generative AI with Python (Practice) / Python 生成式 AI（实践）

Language: [English](#english) | [中文](#中文)

<a id="english"></a>

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

---

<a id="中文"></a>

# Python 生成式 AI（实践）

语言： [English](#english) | [中文](#中文)

这是一个动手实践、课程风格的实验仓库，用于构建 Python 生成式 AI 应用。

本仓库聚焦以下方向的实战示例：
- 预训练模型，
- 托管与本地大语言模型集成，
- 提示词工程模式，
- 向量嵌入/向量数据库，
- 检索增强生成（RAG），以及
- 智能体工作流。

## 项目结构

- `c02_pretrained_model/`：基于 Transformer 的任务（NER、QA、摘要、翻译、文生图/音频等）
- `c03_large_language_model/`：LLM 服务商集成与 LangChain 模式（路由、并行链、安全/话题控制）
- `c04_prompt_engineering/`：少样本、自一致性、提示链、自反馈
- `c05_vector_databases/`：分块、嵌入、加载、存储、检索、综合应用
- `c06_RAG/`：BM25/TF-IDF、混合检索、查询扩展、RAG 评估
- `c07_agentic_system/`：智能体式 RAG 实验
- `miscellany/`：工具脚本与独立实验
- `tests/`：与核心模块对应的 pytest 测试套件

> 注意：部分目录/文件名由于课程演进原因，保留了拼写错误（例如 `data_loder/`、`sematic_chunking.py`、`structed_chunking.py`）。

## 环境要求

- Python `>=3.11`
- 推荐使用 `uv` 进行依赖与环境管理

## 环境配置

### 1) 安装依赖

```bash
uv sync
```

备选方式：

```bash
pip install -e .
```

### 2) 配置环境变量

在仓库根目录创建 `.env`。常见环境变量包括：

```env
OPENAI_API_KEY=...
GROQ_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
HUGGINGFACEHUB_API_TOKEN=...
PINECONE_API_KEY=...
```

许多脚本会调用 `load_dotenv(find_dotenv(usecwd=True))`，因此将 `.env` 放在根目录很重要。

## 运行示例

在仓库根目录执行：

```bash
uv run python c03_large_language_model/chat_with_openai.py
uv run python c05_vector_databases/chunking/fixed_size.py
uv run python c06_RAG/simple_RAG.py
```

大多数示例脚本是带 `# %%` 单元的 notebook 风格 Python 文件，也适合在 VS Code 中分步执行。

## 测试

运行完整测试套件：

```bash
uv run pytest
```

运行指定模块测试：

```bash
uv run pytest tests/c05_vector_databases/chunking
```

可选语法检查：

```bash
uv run python -m compileall .
```

## 开发说明

- 尽量对重量级模型采用惰性导入。
- 新示例请添加到对应的 `c0X_` 模块。
- 保留 `# %%` / `# %% [markdown]` 单元结构，便于教程式执行。
- 修改可复用工具时，请在 `tests/` 中新增或更新测试。

## 依赖说明

核心依赖在 `pyproject.toml` 中管理，主要包括：
- LangChain 生态（`langchain-*`）
- 模型与工具链（`transformers`、`torch`、`tensorflow`、`diffusers`）
- 检索/向量技术栈（`chromadb`、`pinecone`、`sentence-transformers`、`rank-bm25`）
- 评估/通用工具（`ragas`、`pytest`、`datasets`、`wikipedia`）

## 许可

仅用于教育/实践。

---

最后更新：2026 年 3 月
