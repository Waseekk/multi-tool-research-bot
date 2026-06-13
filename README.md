---
title: Multi-Agent Tool Calling System
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: "1.49.1"
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---

# Research Intelligence Assistant

A production-grade AI research platform built with Streamlit, LangGraph, and LangChain. It combines a multi-provider LLM stack, a 13-tool ReAct agent, PDF paper analysis with semantic search, user authentication, and token-by-token streaming into a single deployable app.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Anthropic](https://img.shields.io/badge/Anthropic_Claude-D97757?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

**Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/waseek/multi-agent-tool-calling)

---

## Architecture

```
+----------------------------------------------------------+
|                      Streamlit UI                        |
|   Hero header  |  Sidebar  |  Chat  |  PDF uploader     |
+----------------------------------------------------------+
         |
         v
+----------------------------------------------------------+
|                     Auth Gate                            |
|  SQLite + bcrypt  |  20 chats/day limit  |  Admin bypass|
+----------------------------------------------------------+
         |
         v
+----------------------------------------------------------+
|               LangGraph ReAct Graph                      |
|                                                          |
|  START                                                   |
|    |                                                     |
|    v                                                     |
|  conversation_manager                                    |
|    - trims history to last N messages                    |
|    - builds rolling conversation summary                 |
|    |                                                     |
|    v                                                     |
|  context_llm  <-----------------------+                  |
|    - selects model by task type       |                  |
|    - rebuilds system prompt per turn  |                  |
|    - binds all 13 tools               |                  |
|    |                                  |                  |
|    +-- tool call? --> enhanced_tools -+                  |
|    |                  (runs tool, returns ToolMessage)   |
|    |                                                     |
|    +-- final answer --> END (streamed token by token)    |
+----------------------------------------------------------+
         |                        |
         v                        v
+-------------------+    +-------------------+
|   LLM Providers   |    |   Tool Layer      |
|                   |    |                   |
|  1. Anthropic     |    |  Academic Search  |
|     Claude Opus   |    |  - ArXiv          |
|     Claude Sonnet |    |  - PubMed         |
|     Claude Haiku  |    |  - Semantic Scholar|
|                   |    |  - OpenAlex       |
|  2. OpenAI        |    |  - Find Related   |
|     GPT-4o        |    |                   |
|     GPT-4o Mini   |    |  Web & General    |
|                   |    |  - DuckDuckGo     |
|  3. Groq          |    |  - Tavily         |
|     Llama 3.3 70B |    |  - Wikipedia      |
|     Llama 4 Scout |    |                   |
|     Llama 3.1 8B  |    |  PDF + Utilities  |
|                   |    |  - pdf_search     |
|  Auto fallback    |    |  - Calculator     |
|  on rate limits   |    |  - Code Analyzer  |
+-------------------+    |  - Weather Info   |
                         |  - File Generator |
                         +-------------------+
                                  |
                                  v
                         +-------------------+
                         |   Data Layer      |
                         |                   |
                         |  ChromaDB         |
                         |  (vector store)   |
                         |  sentence-        |
                         |  transformers     |
                         |  embeddings       |
                         |                   |
                         |  SQLite           |
                         |  (users + usage)  |
                         +-------------------+
```

---

## Features

- **Multi-provider LLM** - Anthropic Claude Opus/Sonnet/Haiku, OpenAI GPT-4o/Mini, Groq Llama 3.3/4/3.1 with automatic failover on rate limits
- **Live model selector** - switch between any available model mid-conversation from the sidebar without losing history
- **13 research tools** - academic databases, web search, PDF RAG, calculator, code analysis, and more
- **PDF paper analysis** - upload research papers, ask questions, get page-level citations; powered by ChromaDB and sentence-transformers
- **User auth** - register/login with bcrypt-hashed passwords, 20 chats/day limit, admin bypass
- **Token-by-token streaming** - tool progress indicators appear inline before the final answer streams
- **Tool badges** - colored pill badges show which tools were used in each response
- **Copy button** - small icon copies the full response to clipboard (works inside Streamlit's iframe sandbox)
- **Conditional references** - structured citations appended only when tools were used or factual claims were made

---

## Tools

### Academic Research
| Tool | Source | Best for |
|---|---|---|
| arxiv | arxiv.org | CS, physics, math, engineering preprints |
| pub_med | NCBI PubMed | Biomedical and life sciences |
| semantic_scholar_search | Semantic Scholar | Citation counts, cross-field impact |
| openalex_search | OpenAlex | Open-access works across all disciplines |
| find_related_papers | arXiv + Semantic Scholar | Related papers on a topic, searched in parallel |
| pdf_search | Uploaded PDFs (ChromaDB) | Semantic search within your own papers |

### Web and General
| Tool | Source | Best for |
|---|---|---|
| wikipedia | Wikipedia API | Definitions, background, factual summaries |
| duckduckgo_search | DuckDuckGo | Current news and real-time web results |
| tavily_search_results_json | Tavily API | AI-optimized web search (requires API key) |

### Utilities
| Tool | Description |
|---|---|
| calculator | Safe eval of math expressions - no hallucinated arithmetic |
| code_analyzer | Syntax checking and basic static analysis |
| weather_info | Current conditions by city (demo data) |
| file_content_generator | Generates CSV, JSON, Python, or Markdown content |

---

## LLM Provider Chain

The system auto-detects available API keys and builds the fallback chain at startup:

```
If ANTHROPIC_API_KEY is set:
  1. claude-opus-4-8        (primary   - most capable, best research reasoning)
  2. claude-sonnet-4-6      (secondary - fast, excellent quality)
  3. gpt-4o-mini            (fallback  - if OPENAI_API_KEY also set)
  4. llama-3.3-70b          (fallback  - if GROQ_API_KEY also set)

If only OPENAI_API_KEY is set:
  1. gpt-4o                 (primary)
  2. gpt-4o-mini            (secondary)
  3. llama-3.3-70b          (fallback  - if GROQ_API_KEY also set)

If only GROQ_API_KEY is set:
  1. llama-3.3-70b-versatile                    (primary)
  2. meta-llama/llama-4-scout-17b-16e-instruct  (secondary)
  3. llama-3.1-8b-instant                       (fallback)
```

Failed models enter a 60-second cooldown before being retried. Task routing selects temperature automatically: `0.0` for math and code, `0.1` for research and analysis, `0.7` for creative tasks.

---

## Project Structure

```
Multi-tool Research Bot Assistant/
├── app.py                  # Streamlit UI, LangGraph graph assembly, streaming
├── requirements.txt
├── .streamlit/
│   ├── config.toml         # Theme, server settings
│   └── secrets.toml        # API keys for cloud deployment (gitignored)
├── data/                   # Auto-created at runtime (gitignored)
│   ├── auth.db             # SQLite: users and daily usage
│   └── chroma_db/          # ChromaDB: PDF vector embeddings
└── src/
    ├── __init__.py
    ├── models.py           # EnhancedLLM (multi-provider failover), ConversationState
    ├── nodes.py            # LangGraph nodes: context_llm, enhanced_tools, task routing
    ├── tools.py            # All 13 tool definitions, contextvars for user/PDF context
    ├── conversation.py     # History trimming and rolling conversation summary
    ├── auth.py             # SQLite auth: register, login, daily limit enforcement
    └── rag.py              # PDFProcessor (PyMuPDF) + ResearchVectorStore (ChromaDB)
```

---

## Getting Started

**Requirements:** Python 3.11+, at least one API key (Anthropic, OpenAI, or Groq)

```bash
git clone https://github.com/Waseekk/multi-tool-research-bot.git
cd multi-tool-research-bot

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file (include whichever keys you have):

```env
ANTHROPIC_API_KEY=sk-ant-...        # Recommended - Claude Opus 4.8
OPENAI_API_KEY=sk-proj-...          # Optional - GPT-4o fallback
GROQ_API_KEY=gsk_...                # Optional - free tier fallback
TAVILY_API_KEY=tvly-...             # Optional - enhanced web search
ADMIN_PASSWORD=your_admin_password  # Password for the admin account
```

Run:

```bash
streamlit run app.py
```

The first PDF upload downloads the embedding model (~420 MB, one-time only). Subsequent uploads are fast.

---

## Deployment

### Hugging Face Spaces
1. Push to a Hugging Face Space repository
2. Add API keys under Settings > Repository secrets

### Streamlit Cloud
1. Push to GitHub
2. Connect at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Add API keys under Settings > Secrets

### Docker
```bash
docker build -t research-bot .
docker run -p 8501:8501 --env-file .env research-bot
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent framework | LangGraph (ReAct loop) |
| LLM orchestration | LangChain |
| LLM providers | Anthropic, OpenAI, Groq |
| PDF processing | PyMuPDF |
| Vector store | ChromaDB (persistent, on-disk) |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Auth | SQLite + bcrypt |
| Web search | DuckDuckGo, Tavily |
| Academic APIs | arXiv, PubMed, Semantic Scholar, OpenAlex |

---

## License

MIT
