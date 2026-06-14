# Detailed Code Documentation — Multi-Tool Research Bot Assistant

**Version:** 2.0  
**Last Updated:** June 2026  
**Stack:** Streamlit · LangGraph · LangChain · Anthropic/OpenAI/Groq · ChromaDB · SQLite · bcrypt

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [File Responsibilities](#3-file-responsibilities)
4. [app.py — Entry Point](#4-apppy--entry-point)
5. [src/auth.py — Authentication](#5-srcauthpy--authentication)
6. [src/models.py — LLM Management & State](#6-srcmodelspy--llm-management--state)
7. [src/nodes.py — LangGraph Nodes](#7-srcnodespy--langgraph-nodes)
8. [src/tools.py — Tool Definitions](#8-srctoolspy--tool-definitions)
9. [src/rag.py — PDF Processing & Vector Search](#9-srcragpy--pdf-processing--vector-search)
10. [src/conversation.py — History Management](#10-srcconversationpy--history-management)
11. [src/logger.py — Logging](#11-srcloggerpy--logging)
12. [LangGraph State Machine](#12-langgraph-state-machine)
13. [Google OAuth PKCE Flow](#13-google-oauth-pkce-flow)
14. [Streaming Architecture](#14-streaming-architecture)
15. [Error Handling & Fallbacks](#15-error-handling--fallbacks)
16. [Key Gotchas](#16-key-gotchas)

---

## 1. System Overview

A production-grade multi-agent research assistant. The entry point is `app.py` (Streamlit).
An authenticated user's message flows through a LangGraph graph where a keyword-based
supervisor routes it to one of two specialized agents, each with its own tool set and
system prompt. Responses stream token-by-token to the browser.

**Key design properties:**
- Stateful per browser tab: `MemorySaver` + unique `thread_id` UUID per session
- Stateless across users: no shared mutable state
- Multi-provider LLM: auto-detects from env vars, falls back automatically on failure
- Per-user PDF isolation: single ChromaDB collection, filtered by `user_id` on every query

---

## 2. Architecture Diagram

```
Browser ──WebSocket──► Streamlit (app.py)
                              │
                    StreamlitChatbot.stream_response()
                              │
                    graph.stream(state, config, stream_mode="messages")
                              │
              ┌───────────────▼───────────────────────┐
              │        LangGraph Compiled Graph         │
              │                                         │
              │  START → conversation_manager           │
              │                   │                     │
              │           supervisor_node               │
              │           (keyword router)              │
              │              │         │                │
              │    research_agent    pdf_agent          │
              │    ↕ research_tools  ↕ pdf_tools        │
              │    (ReAct loop)      (ReAct loop)       │
              │              │         │                │
              │              └────►END◄┘                │
              └─────────────────────────────────────────┘
                    │               │              │
              LLM Providers    Tool APIs      Data Layer
              Anthropic         arXiv etc.    SQLite (auth)
              OpenAI                          ChromaDB (PDFs)
              Groq                            MemorySaver (conv)
```

---

## 3. File Responsibilities

| File | Responsibility |
|------|---------------|
| `app.py` | Streamlit UI, auth gate, Google OAuth, chatbot class, graph wiring, streaming |
| `src/auth.py` | SQLite schema, bcrypt register/login, Google OAuth upsert, daily limit |
| `src/models.py` | `ConversationState` TypedDict, `EnhancedLLM` multi-provider manager |
| `src/nodes.py` | `supervisor_node`, `research_agent`, `pdf_agent`, `_make_llm_node` factory |
| `src/tools.py` | All 13 `@tool` functions, `initialize_tools()` factory, ContextVars |
| `src/rag.py` | `PDFProcessor` (PyMuPDF), `ResearchVectorStore` (ChromaDB) |
| `src/conversation.py` | `ConversationManager`: trim history, rolling summary |
| `src/logger.py` | `get_logger(name)`: rotating file + console handler factory |
| `scripts/update_claude_md.py` | Auto-regenerates CLAUDE.md File Map after edits (PostToolUse hook) |

---

## 4. app.py — Entry Point

### Module-level startup
```python
load_dotenv()                    # reads .env
# copy st.secrets → os.environ  # for Streamlit Cloud / HF Spaces
init_db()                        # create SQLite tables, seed admin
st.set_page_config(...)
```

### TOOL_COLORS dict
Maps tool name → hex color. Used by `render_tool_badges()` to render colored pills
after each response. Add an entry here when adding a new tool.

### StreamlitChatbot class

```python
class StreamlitChatbot:
    # Stored in st.session_state["chatbot"] — survives reruns without
    # rebuilding the graph (which would lose MemorySaver history)

    def __init__(self):
        # On first run: build everything, store in session_state
        # On reruns: restore from session_state (no rebuild)

    def _build_graph(self) -> CompiledGraph:
        # Wires: conversation_manager → supervisor → [research_agent, pdf_agent]
        # Each agent has its own isolated ToolNode
        # Compiled with MemorySaver checkpointer

    def stream_response(self, message, thread_id):
        # Sets ContextVars (_active_user_id, _active_pdf_names)
        # graph.stream(state, config, stream_mode="messages")
        # Yields str chunks: tool progress lines + AI text tokens
        # Handles Anthropic (list of blocks) vs OpenAI/Groq (plain string) content

    def chat(self, message, thread_id):
        # Non-streaming invoke for sidebar example buttons
```

### Auth page (`_show_auth_page`)
Renders Google OAuth button (if credentials configured) + Login/Register tabs.
Google redirect handling is inline: checks `st.query_params.get("code")` on every
rerun — if present, completes the OAuth exchange and sets `logged_in_user`.

### Main app flow (`main()`)
```
1. Auth gate: if no logged_in_user → _show_auth_page(); st.stop()
2. Resolve API keys: user-entered key > .env key (for each provider)
3. Write resolved keys to os.environ so LangChain SDKs pick them up
4. Build/restore StreamlitChatbot
5. Sidebar: account info, daily limit, model selector, API keys, PDFs, tools
6. PDF upload handler: PDFProcessor → ResearchVectorStore.add_pdf()
7. Display chat history from st.session_state.messages
8. Chat input → daily limit check → ContextVar.set() → stream_response()
9. Post-response: increment_chat_count, tool badges, copy button
```

---

## 5. src/auth.py — Authentication

### Database schema
```sql
users (id, email, password_hash TEXT, is_admin INTEGER, created_at)
daily_usage (id, user_id, date TEXT, chat_count INTEGER, UNIQUE(user_id, date))
```
`password_hash` is an empty string for Google OAuth users (they can't use email login).

### Key functions

| Function | Description |
|----------|-------------|
| `init_db()` | CREATE IF NOT EXISTS + seed admin from ADMIN_PASSWORD env var. Idempotent. |
| `register_user(email, password)` | Validate, bcrypt hash, INSERT. Returns (bool, error_msg). |
| `login_user(email, password)` | Lookup + bcrypt.checkpw. Returns (bool, user_dict). |
| `login_or_create_google_user(email, name)` | Upsert: returns existing or creates new with empty hash. |
| `is_daily_limit_reached(user_id, is_admin)` | False for admins; else `chat_count >= DAILY_LIMIT (20)`. |
| `increment_chat_count(user_id)` | INSERT OR IGNORE + UPDATE (upsert pattern for today's row). |

### bcrypt details
`bcrypt.hashpw(password.encode(), bcrypt.gensalt())` — work factor 12 (gensalt default).
`bcrypt.checkpw()` is constant-time (prevents timing attacks).

---

## 6. src/models.py — LLM Management & State

### ConversationState (TypedDict)
LangGraph shared state. `messages` uses `add_messages` reducer (appends, not replaces).
All other fields use last-write-wins. Key fields:
- `current_agent: str` — "research" | "pdf", set by supervisor
- `active_pdfs: List[str]` — PDF names for this session, set by supervisor from ContextVar
- `error_count: int` — incremented per LLM/tool failure, for future circuit-breaker logic

### ModelConfig
Runtime health stats per model: `failure_count`, `last_failure_time`, `success_count`.
Used by `_is_model_available()` to enforce 60-second cooldown after failure.

### EnhancedLLM

**Provider chain construction** (`__init__`):
1. Check `LLM_PROVIDER` env var — if set, masks other keys to force that provider
2. Detect keys in priority: ANTHROPIC > OPENAI > GROQ
3. Build `primary_config`, `secondary_config`, `fallback_configs`

**Key methods:**
- `get_llm(prefer_secondary, force_model)` — walks chain respecting cooldowns
- `get_model_for_task(task_type)` — keyword → temperature routing, respects `user_forced_model`
- `_create_llm_instance(config, temperature)` — dispatches to ChatAnthropic/ChatOpenAI/ChatGroq
- `get_model_stats()` — returns health dict for sidebar display

**TASK_TEMPERATURES:**
```python
math/coding  → 0.0   (deterministic)
analysis     → 0.05
reasoning    → 0.1
general      → 0.1
creative     → 0.7
```

---

## 7. src/nodes.py — LangGraph Nodes

### supervisor_node(state) → dict
Keyword routing, no LLM call. Reads last `HumanMessage`, checks `_active_pdf_names` ContextVar.
Returns `{"current_agent": "research"|"pdf", "active_pdfs": [...]}`.

### route_after_supervisor(state) → str
Conditional edge function: `state["current_agent"] + "_agent"`.

### _make_llm_node(tools, llm_manager, get_prompt, agent_label) → Callable
Factory that returns a LangGraph node function. Per-call behavior:
1. Get task-optimized LLM from `get_task_optimized_llm()`
2. Build pdf_ctx string from `_active_pdf_names` ContextVar
3. Build system prompt via `get_prompt(state, pdf_ctx)`
4. Strip existing `SystemMessage`, prepend fresh one (prompt always current, not frozen)
5. `llm.bind_tools(tools).invoke(full_messages)` → return `{"messages": [response]}`
6. On exception: walk fallback chain (skip failed model, trim history for Groq 8B)

### _make_tool_node(tools) → Callable
Wraps `ToolNode(tools)`. On exception: generates recovery `ToolMessage` per tool call ID
so the agent can try a different approach rather than crashing.

### create_agent_nodes(tools, llm_manager) → tuple
Public factory used by `app.py::_build_graph()`. Splits tools:
```python
research_tools = [t for t in tools if t.name not in {"pdf_search"}]
pdf_tools      = [t for t in tools if t.name in {"pdf_search", "arxiv",
                  "semantic_scholar_search", "find_related_papers"}]
```

### get_task_optimized_llm(llm_manager, user_message)
Keyword classifier → `llm_manager.get_model_for_task(task_type)`. Intentionally simple
(keyword matching) — an LLM-based classifier would add a full round-trip just to route.

---

## 8. src/tools.py — Tool Definitions

### ContextVars (module-level)
```python
_active_user_id:   ContextVar[str]   # set by app.py before each graph.stream()
_active_pdf_names: ContextVar[list]  # same
```
Why ContextVars instead of session_state: LangGraph's `ToolNode` runs in a callback
context where `st.session_state` is not reliably accessible across threads.

### Tool inventory

| Tool | Source | Key note |
|------|--------|----------|
| `arxiv` | LangChain wrapper | 5 results × 3000 chars (~4k tokens) |
| `wikipedia` | LangChain wrapper | 2 results × 800 chars (articles are huge) |
| `pub_med` | LangChain wrapper | 4 results × 1200 chars, requires `xmltodict` |
| `duckduckgo_search` | LangChain wrapper | Free, no key, always-on web fallback |
| `tavily_search_results_json` | LangChain wrapper | Paid, only added if TAVILY_API_KEY set |
| `semantic_scholar_search` | Custom REST | `limit` param removed from signature (Groq schema rejects string coercion) |
| `openalex_search` | Custom REST | Contact email in User-Agent (ToS requirement) |
| `find_related_papers` | Custom | Parallel arXiv + Semantic Scholar via ThreadPoolExecutor |
| `pdf_search` | Custom RAG | Reads `_active_user_id` ContextVar; k=4 chunks × 600 chars |
| `calculator` | Custom | Character whitelist (not eval sandbox); safe for simple arithmetic |
| `code_analyzer` | Custom | `compile()` for syntax check; no execution |
| `weather_info` | Custom | Returns simulated demo data |
| `file_content_generator` | Custom | Templates for csv/json/python/markdown |

### initialize_tools() → List
Constructs tool list in this order: arxiv, wikipedia, pubmed, [tavily if key], duckduckgo,
calculator, code_analyzer, weather_info, file_content_generator, semantic_scholar,
openalex, find_related_papers, [pdf_search if RAG deps installed].

---

## 9. src/rag.py — PDF Processing & Vector Search

### PDFProcessor.process_pdf(file_bytes, filename) → dict
- Opens with `fitz.open(stream=file_bytes, filetype="pdf")`
- Extracts text per page, inserts `[Page N]` markers
- Returns `{text, pages_content: {page: text}, metadata: {title, authors, pages}, filename}`
- Title/authors from PDF metadata; falls back to filename if missing

### ResearchVectorStore

**Initialization:**
```python
client     = chromadb.PersistentClient(path="data/chroma_db")
embed_fn   = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
collection = client.get_or_create_collection(
    name="research_papers",
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"},  # cosine better than L2 for high-dim text
)
splitter   = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " "]
)
```
Larger chunks (1500 vs typical 1000) because research papers have dense paragraphs.

**add_pdf(pdf_data, user_id) → int:**
- Delete stale chunks: `collection.get(where={...})` + `collection.delete(ids=...)`
- Chunk → embed → `collection.add(ids, documents, metadatas)`
- Chunk IDs: `{user_id}_{safe_filename}_{index}` (unique, no spaces)
- Metadata per chunk: `{user_id, pdf_name, paper_title, chunk_id, page_num}`

**search(query, user_id, k=4) → List[dict]:**
- Count user docs first to avoid `n_results=0` error (ChromaDB raises on empty set)
- `collection.query(query_texts=[query], n_results=k, where={"user_id": user_id})`
- **The `where` filter is the only isolation mechanism** — removing it exposes all users' PDFs
- Distance → score: `round(max(0.0, 1.0 - dist / 2), 3)` (cosine [0,2] → similarity [0,1])

---

## 10. src/conversation.py — History Management

### ConversationManager

**trim_history(messages, max_history=20) → List:**
Keeps the last `max_history` messages. Simple but effective — LangGraph accumulates
messages indefinitely via `add_messages` reducer, so trimming is necessary to prevent
context overflow on long sessions.

**summarize_conversation(messages) → str:**
Scans last 3 `HumanMessage` contents, joins them into "User asked about: X; Y; Z".
Pure Python, no LLM call. This summary is injected into every system prompt so agents
have context without seeing the full history.

---

## 11. src/logger.py — Logging

`get_logger(name)` is the only public function. Call with `__name__` in each module.

**Output:**
- Console: `INFO` and above
- `logs/app.log`: `DEBUG` and above, RotatingFileHandler (5 MB max, 3 backups)

**Guard against duplicate handlers:** checks `logger.handlers` before attaching.
Important in Streamlit because modules can be re-imported on each rerun.

**Format:** `2026-06-15 12:00:00 [INFO    ] src.nodes: message`

---

## 12. LangGraph State Machine

```
START
  │
  ▼
conversation_manager    (trims history, builds rolling summary)
  │
  ▼
supervisor_node         (keyword route: research | pdf)
  │                │
  ▼                ▼
research_agent   pdf_agent
  │                │
  ▼                ▼
research_tools   pdf_tools    (ToolNode — dispatches tool calls)
  │                │
  └────────────────┘
         │  (loop until no tool_calls)
         ▼
        END
```

**Conditional edges:**
- `supervisor → route_after_supervisor` → `research_agent` | `pdf_agent`
- `research_agent → tools_condition` → `research_tools` | `__end__`
- `pdf_agent → tools_condition` → `pdf_tools` | `__end__`

**MemorySaver:** `thread_id` (UUID per browser tab) keys the checkpoint store.
Each `graph.stream()` call with the same `thread_id` continues the conversation.
A new tab gets a new UUID → isolated conversation.

**recursion_limit=12:** allows up to 5 tool cycles per response.
`1 (conv_manager) + 1 (supervisor) + 2×N (agent+tools) + 1 (final answer) ≤ 12 → N ≤ 4.5`

---

## 13. Google OAuth PKCE Flow

PKCE (Proof Key for Code Exchange) prevents authorization code interception.
Streamlit's complication: session state is lost when the browser navigates to
Google and back. Solution: embed the PKCE verifier in the `state` parameter.

```
1. _google_auth_url()
   verifier  = secrets.token_urlsafe(32)                     # random secret
   challenge = base64url(sha256(verifier))                   # S256 transform
   state     = base64url(verifier)                           # encode for URL safety
   redirect → Google with code_challenge + state

2. Google consent screen → user approves

3. Google redirects back to app with ?code=AUTH_CODE&state=OUR_STATE

4. _google_exchange_code(code, state)
   verifier = base64url_decode(state)                        # recover from state
   POST /token {code, code_verifier=verifier, ...}           # PKCE verification
   GET /userinfo with access_token

5. login_or_create_google_user(email, name) → user dict
   st.session_state.logged_in_user = user dict
   st.rerun()
```

**Why `state` instead of `session_state`:** Streamlit creates a new WebSocket
connection after the OAuth redirect. Any `session_state` written before the redirect
is gone. Google echoes `state` back in the redirect URL, so it survives.

---

## 14. Streaming Architecture

`StreamlitChatbot.stream_response()` is a Python generator consumed by `st.write_stream()`.

```python
for chunk, metadata in self.graph.stream(state, config, stream_mode="messages"):

    node = metadata.get("langgraph_node", "")
    if node in ("research_agent", "pdf_agent") and node not in announced:
        yield f"*Using **{label}**...*\n\n"   # one-time agent announcement
        announced.add(node)

    if isinstance(chunk, ToolMessage):
        yield f"\n*Searching with **{chunk.name}**...*\n\n"

    elif isinstance(chunk, AIMessageChunk):
        content = chunk.content
        if isinstance(content, list):
            # Anthropic: [{'type': 'text', 'text': '...', 'index': 0}]
            text = "".join(b.get("text","") for b in content if b.get("type")=="text")
        else:
            # OpenAI / Groq: plain string
            text = content or ""
        if text:
            yield text
```

**Why Anthropic returns a list:** Anthropic supports mixed content blocks (text,
tool_use, image) in a single response. LangChain preserves this structure in
streaming. Index field is used to reassemble parallel tool call streams.

---

## 15. Error Handling & Fallbacks

### LLM failure (nodes.py)
1. Log error
2. Identify failed model name
3. Walk `[primary, secondary] + fallback_configs` in order
4. Skip failed model; try each in turn
5. If Groq 8B (`max_tokens <= 2048`): trim messages to `[:1] + [-3:]` before invoking
6. On success: update `llm_manager.current_config`, return response
7. If all fail: return user-friendly `AIMessage` ("rate limited" or "all models failed")

### Tool failure (nodes.py `_make_tool_node`)
`ToolNode.invoke()` failure → generate `ToolMessage(content="Tool error: ...",
tool_call_id=tc["id"])` for each pending tool call. The agent sees the error and
can try a different approach on the next loop iteration.

### LLM cooldown (models.py)
`ModelConfig.last_failure_time` — if set and within 60 seconds, `_is_model_available()`
returns False. The fallback chain skips unavailable models automatically.

---

## 16. Key Gotchas

**ContextVars vs session_state:** Never try to read `st.session_state` inside a
LangGraph tool. Use `_active_user_id.get()` and `_active_pdf_names.get()` instead.

**ChromaDB user isolation:** The `where={"user_id": user_id}` filter in `search()`
is the only thing preventing cross-user data leakage. Do not remove it.

**Anthropic streaming content:** Always check `isinstance(chunk.content, list)` in
streaming code. Assuming a string will silently discard Anthropic responses.

**Google OAuth state → verifier:** The PKCE verifier is base64url-encoded without
padding (`=`). When decoding, add padding back: `state + "=" * (-len(state) % 4)`.

**Groq 8B TPM cap:** `llama-3.1-8b-instant` has a 6000 tokens/minute hard cap.
Trim history to `messages[:1] + messages[-3:]` before invoking it. Larger Groq
models don't have this issue.

**MemorySaver is in-process:** Conversation history is lost on app restart. This is
by design (no DB overhead). The rolling summary in `conversation_summary` provides
continuity across context trimming but not across restarts.

**Streamlit reruns:** Every user interaction triggers a full Python rerun. The
`chatbot_initialized` flag in `session_state` guards against rebuilding the graph
on every rerun. Never put expensive initialization outside this guard.
