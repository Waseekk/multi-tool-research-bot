# Technical Documentation: Multi-Tool Research Bot Assistant

**Version:** 2.0  
**Last Updated:** June 2026  
**Tech Stack:** Streamlit · LangGraph · LangChain · Anthropic / OpenAI / Groq · ChromaDB · SQLite

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [File Structure](#3-file-structure)
4. [Core Components](#4-core-components)
5. [Data Flow](#5-data-flow)
6. [Auth System](#6-auth-system)
7. [PDF RAG Pipeline](#7-pdf-rag-pipeline)
8. [LLM Provider Management](#8-llm-provider-management)
9. [Logging](#9-logging)
10. [Configuration & Environment Variables](#10-configuration--environment-variables)
11. [Deployment](#11-deployment)
12. [Extension Guide](#12-extension-guide)

---

## 1. Project Overview

Multi-agent AI research platform. Users authenticate, optionally upload research PDFs,
then chat with a supervisor-routed agent that can:

| Capability | Tools / Component |
|---|---|
| Academic paper search | arxiv, pub_med, semantic_scholar_search, openalex_search |
| Related paper discovery | find_related_papers (parallel arXiv + Semantic Scholar) |
| Web search | duckduckgo_search, tavily_search_results_json |
| Uploaded PDF analysis | pdf_search (ChromaDB RAG, user-isolated) |
| Math & code | calculator, code_analyzer |
| Utilities | weather_info, file_content_generator, wikipedia |
| Multi-provider LLM | Anthropic Claude, OpenAI GPT-4o, Groq Llama (auto-fallback) |
| Auth | Email/password (bcrypt) + Google OAuth (PKCE) |
| Usage limits | 20 chats/day for non-admin users |

---

## 2. Architecture

```
Browser
  │ HTTP (Streamlit WebSocket)
  ▼
┌──────────────────────────────────────────────────────────────┐
│  Streamlit app.py                                            │
│  Auth Gate → if logged in → Sidebar + Chat UI               │
│                StreamlitChatbot.stream_response()            │
└──────────────────────────────────────────────────────────────┘
                          │
                          │ graph.stream(state, config, stream_mode="messages")
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  LangGraph Compiled Graph (MemorySaver checkpointer)         │
│                                                              │
│  START → conversation_manager → supervisor_node              │
│                                       │           │          │
│                               research_agent   pdf_agent     │
│                               ↕ research_tools ↕ pdf_tools  │
│                                       └─────────┘            │
│                                           END                │
└──────────────────────────────────────────────────────────────┘
          │                │                  │
          ▼                ▼                  ▼
   LLM Providers     Tool APIs         Data Layer
   Anthropic Claude  arXiv, PubMed,    SQLite auth.db
   OpenAI GPT-4o     Semantic Scholar, ChromaDB chroma_db/
   Groq Llama        DuckDuckGo, etc.  MemorySaver (in-process)
```

### Multi-Agent Routing

The `supervisor_node` is a fast keyword router (no LLM call). It reads the last
`HumanMessage` and checks `_active_pdf_names` ContextVar:

```
PDF keywords + PDFs loaded  → pdf_agent   (pdf_search, arxiv, scholar, find_related)
Research keywords present   → research_agent (all other 12 tools)
Ambiguous + PDFs loaded     → pdf_agent
Ambiguous + no PDFs         → research_agent
```

---

## 3. File Structure

```
Multi-tool Research Bot Assistant/
├── app.py                    ← Streamlit entry point; auth gate, UI, graph wiring
├── requirements.txt          ← Python dependencies
├── .env                      ← API keys (gitignored)
├── .env.example              ← Template for .env
├── CLAUDE.md                 ← Claude Code project context (gitignored, auto-updated)
├── Dockerfile                ← Container deployment
├── scripts/
│   └── update_claude_md.py   ← Auto-regenerates CLAUDE.md file map on each edit
├── src/
│   ├── auth.py               ← SQLite auth: register, login, daily limits, Google OAuth
│   ├── conversation.py       ← ConversationManager: trim history, rolling summary
│   ├── logger.py             ← Centralized logging (RotatingFileHandler + console)
│   ├── models.py             ← EnhancedLLM (multi-provider fallback), ConversationState
│   ├── nodes.py              ← supervisor_node, research_agent, pdf_agent, tool nodes
│   ├── rag.py                ← PDFProcessor (PyMuPDF) + ResearchVectorStore (ChromaDB)
│   └── tools.py              ← All 13 tools + initialize_tools() factory
├── data/
│   ├── auth.db               ← SQLite (auto-created on first run, gitignored)
│   └── chroma_db/            ← ChromaDB vector store (gitignored)
├── logs/
│   └── app.log               ← Rotating log file (gitignored)
├── .streamlit/
│   ├── config.toml           ← Theme, server settings (tracked)
│   └── secrets.toml          ← API keys for Streamlit Cloud (gitignored)
└── docs/
    ├── HOW_IT_WORKS.md
    ├── TECHNICAL_DOCUMENTATION.md
    └── DETAILED_CODE_DOCUMENTATION.md
```

---

## 4. Core Components

### StreamlitChatbot (`app.py`)

Stored in `st.session_state` so it survives Streamlit reruns without rebuilding
the LangGraph graph (which would lose `MemorySaver` conversation history).

Key methods:
- `_build_graph()` — wires the multi-agent LangGraph graph
- `stream_response(message, thread_id)` — yields string chunks to `st.write_stream()`
- `chat(message, thread_id)` — non-streaming invoke (sidebar example buttons)

### EnhancedLLM (`src/models.py`)

Multi-provider LLM manager. On init, builds a fallback chain from available API keys:

```
ANTHROPIC_API_KEY → claude-opus-4-8 → claude-sonnet-4-6 → (groq fallback)
OPENAI_API_KEY    → gpt-4o → gpt-4o-mini → (groq fallback)
GROQ_API_KEY only → llama-3.3-70b → llama-4-scout → llama-3.1-8b
```

- `LLM_PROVIDER` env var forces a specific provider regardless of other keys
- Failed models enter a 60-second cooldown before being retried
- `get_model_for_task(task_type)` selects temperature based on query type (math=0.0, creative=0.7)

### ConversationState (`src/models.py`)

LangGraph `TypedDict` shared across all nodes:

```python
messages: Annotated[List[AnyMessage], add_messages]  # add_messages reducer appends
conversation_summary: str          # rolling 1-2 sentence summary
current_agent: str                 # "research" | "pdf" — set by supervisor
active_pdfs: List[str]             # PDF names for this user session
error_count: int                   # incremented on LLM failures
last_tool_used: str                # injected into next system prompt
current_model_used: str            # for sidebar display
```

### Agents (`src/nodes.py`)

Both agents use `_make_llm_node()` factory. Per-turn behavior:
1. Get task-optimized LLM (keyword → model + temperature)
2. Build fresh `SystemMessage` (conversation summary + pdf context)
3. Strip old `SystemMessage` from history, prepend new one
4. `llm.bind_tools(agent_tools).invoke(messages)`
5. On failure: walk fallback chain (skip failed model, trim history for Groq 8B)

### ResearchVectorStore (`src/rag.py`)

ChromaDB-backed semantic search. Single shared collection `research_papers`.
Per-user isolation via `where={"user_id": user_id}` on every query.

```
add_pdf(pdf_data, user_id)
  → delete stale chunks (same user + filename)
  → RecursiveCharacterTextSplitter (chunk_size=1500, overlap=300)
  → SentenceTransformer all-mpnet-base-v2 embeddings (768-dim)
  → store with metadata {user_id, pdf_name, paper_title, page_num}

search(query, user_id, k=4)
  → ChromaDB cosine similarity filtered by user_id
  → returns [{text, pdf_name, paper_title, page_num, score}, ...]
```

---

## 5. Data Flow

```
1. User submits message
2. app.py checks daily limit (SQLite)
3. _active_user_id.set() + _active_pdf_names.set()  ← ContextVars for tool isolation
4. graph.stream(initial_state, config, stream_mode="messages")
5. conversation_manager: trim history, build rolling summary
6. supervisor_node: keyword-route to research_agent or pdf_agent
7. Agent: build system prompt → llm.bind_tools().invoke()
8. If tool_calls: tool_node dispatches, returns ToolMessages
9. Agent loops until no tool_calls → END
10. stream_response() yields chunks to st.write_stream()
    - ToolMessage → "*Searching with tool_name...*"
    - AIMessageChunk → extract text (Anthropic: list of blocks; OpenAI/Groq: string)
11. Post-response: increment_chat_count(), render tool badges + copy button
```

---

## 6. Auth System

**SQLite schema** (`data/auth.db`, `src/auth.py`):

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,   -- empty string for Google OAuth users
    is_admin INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE daily_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    date TEXT NOT NULL,            -- YYYY-MM-DD
    chat_count INTEGER DEFAULT 0,
    UNIQUE(user_id, date)
);
```

**Email/password:** bcrypt hash (salt rounds=12). Admin email `waseekirtefa@gmail.com`
auto-seeded from `ADMIN_PASSWORD` env var on `init_db()`.

**Google OAuth (PKCE flow):**
1. `_google_auth_url()` generates PKCE verifier + challenge, encodes verifier into
   `state` param (Streamlit loses session state on browser redirect)
2. Google redirects back with `?code=...&state=...`
3. `_google_exchange_code(code, state)` decodes verifier from `state`, exchanges code
   for access token, fetches user profile
4. `login_or_create_google_user(email, name)` upserts user (empty password_hash)

**Daily limit:** 20 chats/day per non-admin user. Enforced by `is_daily_limit_reached()`.
Resets at midnight (new date = new row in `daily_usage`).

---

## 7. PDF RAG Pipeline

```
Upload → PDFProcessor.process_pdf()
  - PyMuPDF extracts text page by page
  - Inserts [Page N] markers for citation tracking
  - Returns {text, pages_content, metadata{title, authors, pages}}

→ ResearchVectorStore.add_pdf(pdf_data, user_id)
  - Deletes stale chunks for same (user, filename)
  - Splits text: chunk_size=1500, overlap=300
  - Embeds with all-mpnet-base-v2 (~420 MB, cached after first download)
  - Stores in ChromaDB with user_id metadata

Query → pdf_search tool
  - _active_user_id ContextVar provides user isolation
  - search(query, user_id, k=4): cosine similarity, capped at 600 chars/chunk
  - Returns [Paper: title, Page: N] formatted chunks
```

**Why ChromaDB over FAISS:** persistent across restarts, built-in metadata filtering
for user isolation. FAISS would need re-indexing on every app restart.

---

## 8. LLM Provider Management

```
Startup: EnhancedLLM.__init__()
  1. Check LLM_PROVIDER env var (optional override)
  2. Detect available keys: ANTHROPIC > OPENAI > GROQ (priority)
  3. Build primary + secondary + fallback_configs chain

Per request: get_model_for_task(task_type)
  1. If user_forced_model set (sidebar selector) → use that model
  2. Keyword-route: math/coding→primary (temp=0.0), analysis/reasoning→secondary (temp=0.1)
  3. If preferred model in 60s cooldown → fallback chain

On LLM failure (nodes.py):
  1. Log error
  2. Walk all_configs, skip failed model
  3. If Groq 8B: trim history to messages[:1] + messages[-3:] (6000 TPM cap)
  4. If all fail: return user-facing rate-limit or error message
```

---

## 9. Logging

Centralized via `src/logger.py`. All modules call `get_logger(__name__)`.

```
logs/app.log    — RotatingFileHandler: 5 MB max, 3 backups (DEBUG+)
Console         — StreamHandler (INFO+)
Format          — 2026-06-15 12:00:00 [INFO    ] src.nodes: supervisor -> research_agent
```

Log entries to look for when debugging:
- `supervisor -> X_agent (has_pdfs=Y)` — routing decision
- `research_agent: anthropic/claude-opus-4-8` — which model handled the request
- `research_agent: LLM call failed: ...` — fallback trigger
- `GoogleAuth: token exchange failed: ...` — OAuth debug
- `Initialized N tools successfully` — startup health check

---

## 10. Configuration & Environment Variables

Copy `.env.example` to `.env` and fill in values. Never commit `.env`.

```
ANTHROPIC_API_KEY     Claude Opus 4.8 / Sonnet 4.6 (primary if set)
OPENAI_API_KEY        GPT-4o / GPT-4o-mini (secondary if set)
GROQ_API_KEY          Llama 3.3 70B etc. (always-on emergency fallback)
LLM_PROVIDER          optional: "anthropic"|"openai"|"groq" (overrides auto-detect)
TAVILY_API_KEY        paid web search (DuckDuckGo used if absent)
GOOGLE_CLIENT_ID      Google OAuth (optional)
GOOGLE_CLIENT_SECRET  Google OAuth (optional)
GOOGLE_REDIRECT_URI   default: http://localhost:8501/
ADMIN_PASSWORD        seeds waseekirtefa@gmail.com as admin on first run
```

For **Streamlit Cloud / HF Spaces**, add these same keys in the platform's secrets UI.
The `secrets.toml` format:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
GOOGLE_CLIENT_ID  = "..."
```

---

## 11. Deployment

### Local

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env           # fill in API keys
streamlit run app.py
```

First run: downloads `all-mpnet-base-v2` embeddings (~420 MB). Cached afterward.

### Hugging Face Spaces

1. Push code to HF Spaces repo (remote `hf` is already configured)
2. Add all env vars in Settings → Repository secrets
3. Add HF Space URL to Google Cloud Console authorized redirect URIs

### Docker

```bash
docker build -t research-bot .
docker run -p 8501:8501 --env-file .env research-bot
```

**Note on HF Spaces ephemeral storage:** `data/` (SQLite + ChromaDB) is wiped on
restart. For production persistence, use a mounted volume (HF Spaces Pro) or a
cloud database.

---

## 12. Extension Guide

### Add a new tool

```python
# 1. src/tools.py
@tool
def my_tool(query: str) -> str:
    """Description the LLM sees when deciding to call this tool."""
    ...

# 2. Add to initialize_tools() return list
tools_list.append(my_tool)

# 3. app.py — add badge color
TOOL_COLORS["my_tool"] = "#FF5722"

# 4. src/nodes.py — assign to research_tools or PDF_TOOL_NAMES
# research_tools gets everything except pdf_search by default
```

### Add a new LLM provider

In `src/models.py::EnhancedLLM._create_llm_instance()`:
```python
elif config.provider == "my_provider":
    from langchain_myprovider import ChatMyProvider
    return ChatMyProvider(model=config.name, temperature=temp)
```

Then add a detection branch in `__init__()` and a corresponding key in `.env.example`.

### Add a new agent

```python
# 1. src/nodes.py
def _my_prompt(state, pdf_ctx) -> str:
    return "You are a specialist in ..."

my_agent = _make_llm_node(my_tools, llm_manager, _my_prompt, "my_agent")
my_tool_node = _make_tool_node(my_tools)

# 2. app.py::_build_graph()
builder.add_node("my_agent", my_agent)
builder.add_node("my_tools", my_tool_node)
builder.add_conditional_edges("supervisor", route_fn, {"my_agent": "my_agent", ...})
builder.add_conditional_edges("my_agent", tools_condition, {"tools": "my_tools", "__end__": END})
builder.add_edge("my_tools", "my_agent")
```
