---
title: Multi-Tool Research Bot
emoji: 🔬
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.49.1
app_file: app.py
pinned: false
license: mit
---

# Multi-Tool Research Bot

An AI research assistant built with Streamlit, LangChain, and LangGraph. It routes user queries to specialized research APIs and utility tools through a ReAct agent loop, streaming responses token-by-token.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)

**Live Demo:** [multi-tool-research-bot-5pmekpzhczcrdtichq3rw4.streamlit.app](https://multi-tool-research-bot-5pmekpzhczcrdtichq3rw4.streamlit.app/)

---

## Tools

### Academic Research
| Tool | Source | Best for |
|---|---|---|
| ArXiv | arxiv.org | Physics, math, CS, engineering preprints |
| PubMed | NCBI | Biomedical and life sciences |
| Semantic Scholar | semanticscholar.org | Citation counts, cross-field impact |
| OpenAlex | openalex.org | Open-access works across all disciplines |

### General Knowledge & Web
| Tool | Source | Best for |
|---|---|---|
| Wikipedia | Wikipedia API | Definitions, background, factual summaries |
| DuckDuckGo | DuckDuckGo Search | Current news and real-time web results |
| Tavily | Tavily API | AI-optimized web search (optional) |

### Utilities
| Tool | Description |
|---|---|
| Calculator | Safe eval of math expressions — no hallucinated arithmetic |
| Code Analyzer | Syntax checking and basic static analysis |
| Weather Info | Current conditions by city (demo/simulated) |
| File Generator | Generates CSV, JSON, Python, or Markdown content |

---

## Architecture

The app uses a **LangGraph ReAct loop** — a single agent that reasons, calls tools, observes results, and continues until it has enough information to respond.

```
User message
    │
    ▼
conversation_manager   — trims history, builds rolling summary from last 3 questions
    │
    ▼
context_llm            — LLM decides: answer directly, or call one or more tools
    │
    ├── tool call(s) ──► enhanced_tools ──► back to context_llm
    │
    └── final answer ──► streamed to user token-by-token
```

**LLM stack (Groq inference, in fallback order):**
1. `llama-3.3-70b-versatile` — primary
2. `llama-3.1-70b-versatile` — secondary
3. `mixtral-8x7b-32768` — fallback
4. `llama-3.1-8b-instant` — fast fallback
5. `gemma2-9b-it` — last resort

Each task type uses a tuned temperature: `0.0` for math and code, `0.05–0.1` for analysis and research, `0.7` for creative tasks.

**Key design decisions:**
- System prompt is rebuilt on every turn (not cached) so context summary and last-used tool are always current
- Per-session `thread_id` (UUID) gives each browser tab its own isolated conversation memory via LangGraph's `MemorySaver`
- Streaming uses `stream_mode="messages"` — tool-use indicators (`Searching with arxiv...`) appear inline before the final answer streams

---

## Project Structure

```
├── app.py                  # Streamlit UI + StreamlitChatbot class + LangGraph graph assembly
├── src/
│   ├── tools.py            # All 11 tool definitions and initialization
│   ├── models.py           # EnhancedLLM (fallback chain, task temperatures), ConversationState
│   ├── nodes.py            # LangGraph nodes: context_llm, enhanced_tools, task routing
│   └── conversation.py     # History trimming and rolling conversation summary
├── .streamlit/
│   ├── config.toml         # Theme and server config
│   └── secrets.toml        # API keys for Streamlit Cloud (gitignored)
├── Dockerfile              # Container deployment
└── requirements.txt
```

---

## Getting Started

**Prerequisites:** Python 3.9+, a [Groq API key](https://console.groq.com/)

```bash
git clone https://github.com/Waseekk/multi-tool-research-bot.git
cd multi-tool-research-bot

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here   # optional
```

Run:

```bash
streamlit run app.py
```

---

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect at [streamlit.io/cloud](https://streamlit.io/cloud)
3. Add `GROQ_API_KEY` (and optionally `TAVILY_API_KEY`) under **Settings > Secrets**

### Docker
```bash
docker build -t research-bot .
docker run -p 8501:8501 --env-file .env research-bot
```

---

## Known Limitations

- Weather data is simulated — not connected to a live API
- Code analysis covers syntax only, not semantic correctness
- DuckDuckGo may occasionally rate-limit on high-frequency requests
- Conversation memory is in-process (`MemorySaver`) and does not persist across server restarts

---

## License

MIT
