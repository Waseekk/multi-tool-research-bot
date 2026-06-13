# How the Research Intelligence Assistant Works

This document explains the full technical architecture of the app: how a user message
travels through the system, how each component makes decisions, and how the planned
multi-agent upgrade will change the design. Written for developers who want to understand,
extend, or debug the codebase.

---

## Table of Contents

1. [Big Picture](#big-picture)
2. [Request Lifecycle - Step by Step](#request-lifecycle)
3. [LangGraph ReAct Agent Loop](#langgraph-react-agent-loop)
4. [LLM Provider Chain](#llm-provider-chain)
5. [Tool System](#tool-system)
6. [PDF RAG Pipeline](#pdf-rag-pipeline)
7. [Authentication and Usage Limits](#authentication-and-usage-limits)
8. [Streaming Architecture](#streaming-architecture)
9. [Multi-Agent Architecture (Planned)](#multi-agent-architecture-planned)

---

## Big Picture

```
Browser
  |
  | HTTP (Streamlit WebSocket)
  v
+-----------------------------------------------------------+
|  Streamlit App  (app.py)                                  |
|                                                           |
|  Auth Gate --> if logged in --> Chat UI                   |
|                                    |                      |
|                              StreamlitChatbot             |
|                                    |                      |
|                            graph.stream()                 |
+-----------------------------------------------------------+
                                     |
                                     | LangGraph state machine
                                     v
+-----------------------------------------------------------+
|  LangGraph Compiled Graph                                 |
|                                                           |
|  conversation_manager                                     |
|         |                                                 |
|         v                                                 |
|  context_llm  <---------+                                |
|         |               |                                |
|    tool calls?     tool results                          |
|         |               |                                |
|         v               |                                |
|  enhanced_tools ---------+                               |
|         |                                                |
|    final answer --> END                                   |
+-----------------------------------------------------------+
       |            |              |
       v            v              v
  LLM Provider  Tool APIs     Vector Store
  (Anthropic,   (ArXiv,       (ChromaDB +
   OpenAI,       PubMed,       sentence-
   Groq)         Scholar,      transformers)
                 Web, etc.)
```

The key design choice is that the agent is **stateful across turns** (via LangGraph's
`MemorySaver`) but **stateless across users** (each browser tab gets its own `thread_id`
UUID). This gives persistent conversation memory per session without any database writes
for the conversation itself.

---

## Request Lifecycle

Here is what happens from the moment a user presses Enter to when the response finishes
streaming.

```
1. User types a message and presses Enter
   |
   v
2. app.py: _is_inappropriate() check
   - blocklist of explicit words
   - if blocked, show warning and stop
   |
   v
3. app.py: is_daily_limit_reached()
   - queries daily_usage table in SQLite
   - admins are never blocked
   - regular users capped at 20 chats/day
   |
   v
4. app.py: _active_user_id.set() and _active_pdf_names.set()
   - ContextVars (not session_state) because LangGraph's ToolNode
     runs in a callback context where Streamlit session_state is
     not reliably accessible from a different thread
   |
   v
5. StreamlitChatbot.stream_response() called
   - calls self.graph.stream() with stream_mode="messages"
   - recursion_limit=12 (allows up to 5 tool call cycles before
     the graph crashes with GraphRecursionError)
   |
   v
6. LangGraph: conversation_manager node
   - ConversationManager.trim_history(): keeps last N messages
     so the context window never overflows
   - ConversationManager.summarize_conversation(): rolling 1-2
     sentence summary of what has been discussed, injected into
     the system prompt on every turn
   |
   v
7. LangGraph: context_llm node (the main reasoning step)
   - get_task_optimized_llm() keyword-routes to model + temperature
     (e.g. "calculate" -> temp=0.0, "research" -> temp=0.1)
   - if user_forced_model is set (sidebar model selector), that
     overrides task routing entirely
   - builds a fresh system prompt with: conversation summary,
     last tool used, list of uploaded PDF names
   - strips old SystemMessage from history, prepends new one
     (so the system prompt is always current, not frozen at turn 1)
   - llm.bind_tools(tools).invoke(messages)
   - if the LLM decides to call tools, it returns an AIMessage
     with tool_calls populated; LangGraph routes to enhanced_tools
   - if the LLM has enough info to answer, it returns a plain
     AIMessage; LangGraph routes to END
   |
   v
8. LangGraph: enhanced_tools node (only if tools were called)
   - wraps LangGraph's built-in ToolNode
   - dispatches each tool call by name
   - returns ToolMessage results back into state
   - records last_tool_used for the next system prompt
   - on tool crash: returns a graceful ToolMessage with the error
     so the LLM can try a different approach
   |
   (loop back to step 7 until LLM produces a final answer)
   |
   v
9. app.py: stream_response() yields text chunks to st.write_stream
   - ToolMessage -> yields "*Searching with tool_name...*"
   - AIMessageChunk -> extracts text (Anthropic returns content as
     a list of typed blocks; OpenAI/Groq return a plain string;
     handled by checking isinstance(content, list))
   |
   v
10. app.py: post-response
    - cast response to str (st.write_stream may return StreamingOutput)
    - increment_chat_count() writes to SQLite
    - render tool badges (colored pills for each tool used)
    - render_copy_btn() (base64-encoded clipboard injection for
      Streamlit's iframe sandbox)
    - append to st.session_state.messages for chat history display
```

---

## LangGraph ReAct Agent Loop

The agent follows the **ReAct** (Reason + Act) pattern: it thinks, optionally acts
(calls a tool), observes the result, thinks again, and repeats until it can give a
final answer.

```
Human: "Find papers on attention mechanisms"
         |
         v
  [context_llm]
  Reasoning: "This is an academic research query. I should use arxiv."
  Decision: call arxiv("attention mechanism transformer")
         |
         v
  [enhanced_tools]
  Executes: arxiv.run("attention mechanism transformer")
  Returns: ToolMessage with 5 paper summaries
         |
         v
  [context_llm]
  Reasoning: "I now have 5 papers. I can synthesize an answer."
  Decision: write final response (no tool calls)
         |
         v
  [END] -> streamed to user
```

**Why recursion_limit=12?**

LangGraph counts every node visit as one step. The minimum for a single tool call is:

```
conversation_manager (1) + context_llm (2) + enhanced_tools (3) + context_llm (4) = 4 steps
```

Claude Opus regularly calls 2-3 tools per query (e.g. pdf_search then find_related_papers).
With limit=5, a second tool call would crash before the final answer could be written.
Limit=12 allows up to 5 tool cycles:

```
1 (conv_manager) + 2*N (tool cycles) + 1 (final answer) <= 12
N <= 5 tool calls
```

**Fallback cascade in context_llm:**

If the primary LLM throws (rate limit, timeout, etc.), the node walks the full
fallback chain before giving up:

```python
for config in [primary, secondary] + fallback_configs:
    if config.name == failed_model:
        continue
    try:
        fallback_llm = _create_llm_instance(config)
        # Groq small models have a 6000 TPM cap -- trim history first
        if config.provider == "groq" and config.max_tokens <= 2048:
            msgs_to_send = messages[:1] + messages[-3:]
        response = fallback_llm.bind_tools(tools).invoke(msgs_to_send)
        return response
    except:
        continue
```

A failed model enters a 60-second cooldown via `last_failure_time` before being retried.

---

## LLM Provider Chain

```
Environment variable check at startup (priority order):

ANTHROPIC_API_KEY set?
  YES -> primary:   claude-opus-4-8       (best research reasoning)
         secondary: claude-sonnet-4-6     (fast, still excellent)
         fallbacks: gpt-4o-mini (if OPENAI), llama-3.3-70b (if GROQ)

OPENAI_API_KEY set (no Anthropic)?
  YES -> primary:   gpt-4o
         secondary: gpt-4o-mini
         fallbacks: llama-3.3-70b (if GROQ)

GROQ_API_KEY only?
         primary:   llama-3.3-70b-versatile
         secondary: meta-llama/llama-4-scout-17b-16e-instruct
         fallback:  llama-3.1-8b-instant
```

**Task-based temperature routing** (in `get_task_optimized_llm`, `nodes.py`):

```
Keyword in message         -> Task type  -> Temperature
"calculate", "compute"     -> math       -> 0.0  (fully deterministic)
"code", "python", "debug"  -> coding     -> 0.0
"analyze", "compare"       -> analysis   -> 0.05
"research", "paper"        -> reasoning  -> 0.1
"write", "story", "poem"   -> creative   -> 0.7
(anything else)            -> general    -> 0.1
```

If `user_forced_model` is set (via the sidebar model selector), this routing is bypassed
entirely and the chosen model is used for every message.

**Provider dispatch** in `_create_llm_instance` (`models.py`):

```python
if config.provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=config.name, temperature=temp, max_tokens=...)

elif config.provider == "openai":
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=config.name, temperature=temp, max_tokens=...)

else:  # groq
    from langchain_groq import ChatGroq
    return ChatGroq(model=config.name, temperature=temp, max_tokens=...)
```

Adding a new provider (e.g. Anthropic Bedrock, Cohere) requires only a new `elif` branch
here plus a new key-detection block in `__init__`.

---

## Tool System

All tools live in `src/tools.py`. Each is a standard LangChain `@tool` decorated function.

**How tools are bound to the LLM:**

```python
llm_with_tools = llm.bind_tools(tools)
response = llm_with_tools.invoke(messages)
# response.tool_calls contains [{name, args, id}, ...] if the LLM decided to use tools
```

The LLM sees the tool name, description, and parameter schema (auto-generated from the
function signature and docstring). It decides which tools to call and with what arguments
based purely on the system prompt and the user message.

**Thread-safe context for tools:**

Tools need to know the current user ID (for ChromaDB isolation) and the loaded PDF names
(for the system prompt). These cannot be passed through LangGraph state to tools directly,
and `st.session_state` is not reliably accessible inside LangGraph's ToolNode threads.
The solution is `contextvars.ContextVar`:

```python
# tools.py
_active_user_id   = contextvars.ContextVar("active_user_id",   default="")
_active_pdf_names = contextvars.ContextVar("active_pdf_names", default=[])

# app.py (before each graph.stream() call)
_active_user_id.set(str(user["id"]))
_active_pdf_names.set(list(st.session_state.get("uploaded_pdfs", {}).keys()))
```

ContextVars are propagated into threads by Python's `asyncio` and `concurrent.futures`
automatically, so the tool always sees the correct user context.

**Token budget management:**

Groq has a 100k tokens/day limit and a per-minute limit per model. Tool outputs are
truncated to prevent context explosion:

```
arxiv:              5 results x 3000 chars  (~4k tokens per call)
pubmed:             4 results x 1200 chars  (~2k tokens)
pdf_search:         4 chunks  x 600  chars  (~1.2k tokens)
find_related_papers: 3 results per source   (~0.8k tokens)
```

---

## PDF RAG Pipeline

RAG = Retrieval-Augmented Generation. When a user uploads a PDF, it is processed into
a vector store so the LLM can search it semantically.

```
User uploads PDF
      |
      v
PDFProcessor.process_pdf()          (src/rag.py)
  - PyMuPDF extracts raw text page by page
  - Inserts [Page N] markers between pages
  - Reads title/author metadata from PDF headers
  - Returns: {text, pages_content, metadata}
      |
      v
ResearchVectorStore.add_pdf()       (src/rag.py)
  - RecursiveCharacterTextSplitter
      chunk_size=1500, overlap=300
      separators=["\n\n", "\n", ". ", " "]
      (larger chunks than typical because research papers
       have dense, long paragraphs)
  - sentence-transformers/all-mpnet-base-v2
      generates 768-dimensional embeddings per chunk
      (~420 MB download on first use, cached after)
  - ChromaDB stores chunks with metadata:
      {user_id, pdf_name, page_num, chunk_id, paper_title}
  - user_id isolation: each user's chunks are tagged and
    filtered by user_id so users never see each other's papers
      |
      v
When user asks about their paper:
      |
      v
pdf_search tool runs
  - query is embedded with the same model
  - ChromaDB cosine similarity search: top k=4 chunks
  - each chunk capped at 600 chars
  - returns formatted results with page citations:
    [Paper: title, Page: N]
    {chunk text}
    (Relevance: 0.87)
```

**Why ChromaDB over FAISS?**
ChromaDB is persistent (data/chroma_db/ survives restarts) and has built-in metadata
filtering. FAISS is faster but in-memory only, meaning PDFs would need re-indexing on
every app restart.

**Why all-mpnet-base-v2?**
It produces 768-dim embeddings with strong semantic understanding for scientific text.
Smaller models (all-MiniLM-L6-v2, 384-dim) are faster but lose nuance in technical
language. The 420 MB one-time download cost is worth the quality gain for research use.

---

## Authentication and Usage Limits

```
data/auth.db (SQLite)

Table: users
+----+-------------------------+---------------------+----------+
| id | email                   | password_hash       | is_admin |
+----+-------------------------+---------------------+----------+
|  1 | waseekirtefa@gmail.com  | $2b$12$...          |    1     |
|  2 | user@example.com        | $2b$12$...          |    0     |
+----+-------------------------+---------------------+----------+

Table: daily_usage
+----+---------+------------+------------+
| id | user_id | date       | chat_count |
+----+---------+------------+------------+
|  1 |       2 | 2026-06-14 |      7     |
+----+---------+------------+------------+
```

**Registration flow:**
1. Email validated (must contain @ and a dot after it)
2. Password hashed with bcrypt (salt rounds = 12, work factor adjusts automatically)
3. If email matches `ADMIN_EMAIL`, `is_admin=1` is set automatically
4. `INSERT` into users table; `IntegrityError` on duplicate email

**Login flow:**
1. Look up row by email (`SELECT ... WHERE email = ?`)
2. `bcrypt.checkpw(entered_password, stored_hash)` - constant-time comparison prevents
   timing attacks
3. Return user dict `{id, email, is_admin}` stored in `st.session_state.logged_in_user`

**Why the database resets on Hugging Face Spaces restarts:**
HF Spaces has an ephemeral filesystem - the `data/` directory is not persisted between
deployments. Every restart starts with a fresh `auth.db`. To fix this, you would need
to mount a persistent storage volume (HF Spaces Pro feature) or use a cloud database
(Supabase, PlanetScale, etc.).

**Daily limit enforcement:**
```python
# Before each chat message
if is_daily_limit_reached(user["id"], user["is_admin"]):
    st.warning("You've reached today's limit of 20 chats.")
    st.stop()

# After each successful response
increment_chat_count(user["id"])
```

The count resets automatically at midnight because the `date` column uses
`date.today().isoformat()` - a new day creates a new row in `daily_usage`.

---

## Streaming Architecture

`st.write_stream()` consumes a Python generator. The generator must yield `str` objects.

```python
# app.py: stream_response() generator

for chunk, _ in self.graph.stream(state, config, stream_mode="messages"):

    if isinstance(chunk, ToolMessage):
        # Tool finished - show progress indicator
        yield f"\n*Searching with **{name}**...*\n\n"

    elif isinstance(chunk, AIMessageChunk):
        # LLM is generating text - extract and yield it
        content = chunk.content

        if isinstance(content, list):
            # Anthropic returns: [{'type': 'text', 'text': '...', 'index': 0}]
            # Must extract text blocks - yielding the list silently breaks st.write_stream
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            # OpenAI and Groq return a plain string
            text = content or ""

        if text:
            yield text
```

**Why Anthropic content is a list:**
Anthropic's API can return mixed content blocks in a single response (text + tool_use +
image). LangChain preserves this structure in streaming chunks. For a pure text chunk,
content is `[{'type': 'text', 'text': '...', 'index': 0}]` rather than a plain string.
The index field is used by Anthropic to reassemble parallel tool call streams.

**Copy button in Streamlit iframes:**
`navigator.clipboard.writeText()` is blocked in sandboxed iframes (Streamlit renders
each component in an iframe). The workaround uses the older `document.execCommand('copy')`
with a hidden textarea:

```javascript
var ta = document.createElement('textarea');
ta.value = atob(base64EncodedText);   // base64 avoids quote escaping issues
ta.style.cssText = 'position:fixed;top:-9999px';
document.body.appendChild(ta);
ta.focus(); ta.select();
document.execCommand('copy');
document.body.removeChild(ta);
```

---

## Multi-Agent Architecture (Planned)

**Current state:** The app runs a single-agent ReAct loop. All 13 tools are available to
one agent that handles every query - whether it's a PDF question, academic search, or
web query.

**Why upgrade to multi-agent?**

A single agent with 13 tools has to decide which tool to use on every turn. This leads to:
- Tool confusion: agent sometimes calls arxiv when the user is asking about their PDF
- Inefficiency: the full tool list is bound to every LLM call even when only 2-3 are relevant
- No specialization: the system prompt tries to cover every scenario in one block

A supervisor pattern solves this by routing each query to a specialized sub-agent that
has only the tools it needs and a focused system prompt.

**Planned graph design:**

```
START
  |
  v
conversation_manager
  |
  v
supervisor_node
  - lightweight keyword router (no LLM call - fast)
  - checks: does the query mention PDFs? are PDFs loaded?
  - routes to one of two agents
  |
  +-----> research_agent <--> research_tools_node
  |              |             (arxiv, pubmed, scholar, openalex,
  |         [loop until]        find_related, web, wikipedia,
  |          final answer       calculator, code, weather, file)
  |
  +-----> pdf_agent <--> pdf_tools_node
                 |        (pdf_search, arxiv, semantic_scholar)
            [loop until]
             final answer
```

**Supervisor routing logic (keyword-based, no LLM call):**

```python
def supervisor_node(state):
    msg = last_human_message(state).lower()
    has_pdfs = bool(state.get("active_pdfs"))

    pdf_keywords = ["pdf", "my paper", "this paper", "the document",
                    "uploaded", "what does it say"]
    research_keywords = ["find papers", "latest research", "arxiv",
                         "pubmed", "search for", "recent studies"]

    if has_pdfs and any(kw in msg for kw in pdf_keywords):
        return {"current_agent": "pdf"}
    elif any(kw in msg for kw in research_keywords):
        return {"current_agent": "research"}
    elif has_pdfs:
        # Default to pdf_agent when papers are loaded
        return {"current_agent": "pdf"}
    else:
        return {"current_agent": "research"}
```

**PDF agent system prompt focus (planned):**

```
1. Always call pdf_search first to ground the answer in the uploaded paper
2. Then call arxiv or semantic_scholar to find related external work
3. Connect the paper to the broader research landscape:
   - What prior work does this build on?
   - What problem is solved, and why this approach?
   - What are the open problems and future directions?
4. Cite everything: [Paper: title, Page: N] for uploaded docs,
   (Author et al., Year) for external papers
```

**Research agent system prompt focus (planned):**

```
1. Pick the most relevant 1-2 sources (not all of them)
2. Synthesize findings - do not just summarize API output
3. Explain methodology rationale and research context
4. Identify consensus vs. active debates
5. Suggest future research directions
```

**What changes in the code:**

```
Current code:                      After multi-agent upgrade:

src/nodes.py                       src/nodes.py
  create_enhanced_nodes()            create_enhanced_nodes()
    context_aware_llm                  supervisor_node()
    enhanced_tool_node                 research_agent_node()
                                       pdf_agent_node()
                                       research_tool_node
                                       pdf_tool_node

app.py _build_graph():             app.py _build_graph():
  conversation_manager               conversation_manager
  -> context_llm                     -> supervisor
  <-> enhanced_tools                 -> research_agent <-> research_tools
                                     -> pdf_agent     <-> pdf_tools
```

**State schema additions needed for multi-agent:**

```python
class ConversationState(TypedDict):
    # existing fields ...
    current_agent: str       # "research" | "pdf"
    active_pdfs:   List[str] # PDF names loaded for this user
```

**Implementation estimate:** 3-4 hours. The supervisor node and routing logic are small.
The main work is splitting the system prompt into two focused prompts and wiring the
separate ToolNodes in the graph.

---

## Key Design Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| State machine | LangGraph ReAct | Built-in MemorySaver, streaming, tool routing |
| Conversation memory | MemorySaver (in-process) | No DB writes needed for conversation |
| PDF embeddings | all-mpnet-base-v2 (768-dim) | Better semantic quality for scientific text |
| Vector store | ChromaDB persistent | Survives restarts, metadata filtering built-in |
| Tool context passing | ContextVar (not session_state) | Thread-safe in LangGraph callbacks |
| Auth storage | SQLite | Zero infrastructure, file-based, simple schema |
| Clipboard copy | execCommand via hidden textarea | navigator.clipboard blocked in Streamlit iframes |
| Anthropic streaming | Extract from content list | Anthropic chunks are typed blocks, not plain strings |
| Recursion limit | 12 | Allows 5 tool cycles; 5 was too tight for Claude Opus |
