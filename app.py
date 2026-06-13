"""
app.py
======
Entry point for the Streamlit application.

Responsibilities:
  - Page config and UI layout
  - Builds the LangGraph graph via StreamlitChatbot._build_graph()
  - Exposes chat() (non-streaming) and stream_response() (streaming) interfaces
  - Manages Streamlit session state so the chatbot survives across reruns

Architecture:
  StreamlitChatbot
    ├── _build_graph()         — assembles the LangGraph ReAct loop
    ├── _manage_conversation() — conversation_manager node (trims history, builds summary)
    ├── _initial_state()       — builds the ConversationState dict for each new message
    ├── chat()                 — non-streaming invoke (used by sidebar example buttons)
    └── stream_response()      — streaming generator (used by main chat input)
"""

import streamlit as st
import os
import base64
import uuid
from typing import List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Streamlit Cloud secrets (sets env vars before any module reads them)
if hasattr(st, "secrets"):
    try:
        for key in ("GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
    except Exception:
        pass

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from src.tools import initialize_tools
from src.models import EnhancedLLM, ConversationState
from src.nodes import create_enhanced_nodes
from src.conversation import ConversationManager
from src.auth import (
    init_db, login_user, register_user,
    is_daily_limit_reached, increment_chat_count,
    get_chat_count_today, DAILY_LIMIT,
)
from src.tools import _active_user_id, _active_pdf_names

init_db()

st.set_page_config(
    page_title="Research Intelligence Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — minimal, reliable selectors only
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Reduce default top padding so the hero sits higher */
    .main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

    /* Buttons: rounded corners, smooth hover */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: opacity 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover { opacity: 0.88; }

    /* Chat message cards */
    [data-testid="stChatMessage"] { border-radius: 12px; }

    /* Text inputs */
    .stTextInput input, .stTextArea textarea { border-radius: 8px; }

    /* Expander borders */
    .stExpander { border-radius: 8px; }

    /* Horizontal rules */
    hr { margin: 10px 0; opacity: 0.3; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tool badge colours
# ---------------------------------------------------------------------------
TOOL_COLORS = {
    "arxiv":                      "#2196F3",
    "pub_med":                    "#4CAF50",
    "wikipedia":                  "#FF9800",
    "duckduckgo_search":          "#9C27B0",
    "tavily_search_results_json": "#673AB7",
    "calculator":                 "#F44336",
    "semantic_scholar_search":    "#00BCD4",
    "openalex_search":            "#009688",
    "code_analyzer":              "#795548",
    "weather_info":               "#03A9F4",
    "file_content_generator":     "#607D8B",
    "pdf_search":                 "#E91E63",
    "find_related_papers":        "#FF5722",
}


def render_tool_badges(tool_names: list) -> str:
    """Return HTML for a row of colored pill badges, one per tool used."""
    if not tool_names:
        return ""
    badges = []
    for name in tool_names:
        color = TOOL_COLORS.get(name, "#9E9E9E")
        label = name.replace("_", " ").replace("tavily search results json", "tavily").title()
        badges.append(
            f'<span style="background:{color};color:white;padding:2px 10px;'
            f'border-radius:12px;font-size:0.73em;margin:2px;display:inline-block;'
            f'font-weight:500">{label}</span>'
        )
    return '<div style="margin-top:6px;line-height:2">' + " ".join(badges) + "</div>"


def render_copy_btn(text: str, btn_key: str) -> None:
    """
    Small ⧉ Copy button rendered via st.components.v1.html.

    Uses document.execCommand('copy') with a hidden textarea — this works inside
    Streamlit's iframe sandbox where navigator.clipboard may be restricted.
    Text is base64-encoded to avoid any quoting / escaping issues.
    """
    b64 = base64.b64encode(str(text).encode("utf-8")).decode()
    st.components.v1.html(f"""
    <div style="display:flex;justify-content:flex-end;padding:2px 0 0">
      <button id="cpb_{btn_key}"
        onclick="(function(enc){{
          var ta=document.createElement('textarea');
          ta.value=atob(enc);
          ta.style.cssText='position:fixed;top:-9999px;left:-9999px';
          document.body.appendChild(ta);ta.focus();ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          var b=document.getElementById('cpb_{btn_key}');
          b.textContent='✓ Copied';
          b.style.color='#22c55e';b.style.borderColor='#22c55e';
          setTimeout(function(){{
            b.textContent='⧉ Copy';
            b.style.color='#64748b';b.style.borderColor='#334155';
          }},2000);
        }})('{b64}')"
        title="Copy response to clipboard"
        style="background:transparent;border:1px solid #334155;border-radius:6px;
               color:#64748b;cursor:pointer;font-size:11.5px;padding:3px 10px;
               font-family:system-ui,sans-serif;letter-spacing:0.2px;
               transition:all 0.2s;white-space:nowrap">
        &#x29c9; Copy
      </button>
    </div>
    """, height=32)


# ---------------------------------------------------------------------------
# Chatbot class
# ---------------------------------------------------------------------------

class StreamlitChatbot:
    """
    Wraps the LangGraph compiled graph and exposes chat/stream interfaces to the UI.

    Stored in st.session_state so it survives Streamlit reruns without rebuilding
    the graph (which would lose MemorySaver conversation history).
    """

    def __init__(self):
        if "chatbot_initialized" not in st.session_state:
            with st.spinner("Initializing Research Intelligence Assistant..."):
                self.tools               = initialize_tools()
                self.llm_manager         = EnhancedLLM()
                self.conversation_manager = ConversationManager()
                self.memory              = MemorySaver()
                self.graph               = self._build_graph()
                st.session_state.chatbot_initialized = True
                st.session_state.chatbot = self
        else:
            chatbot                  = st.session_state.chatbot
            self.tools               = chatbot.tools
            self.llm_manager         = chatbot.llm_manager
            self.conversation_manager = chatbot.conversation_manager
            self.memory              = chatbot.memory
            self.graph               = chatbot.graph

    def _build_graph(self) -> StateGraph:
        """
        Assembles the LangGraph ReAct loop:
          conversation_manager → context_llm ⇄ enhanced_tools → END

        tools_condition routes to enhanced_tools when the LLM requests tool calls,
        or to END when it produces a plain text response.
        """
        context_llm, enhanced_tools = create_enhanced_nodes(self.tools, self.llm_manager)

        builder = StateGraph(ConversationState)
        builder.add_node("conversation_manager", self._manage_conversation)
        builder.add_node("context_llm",          context_llm)
        builder.add_node("enhanced_tools",        enhanced_tools)

        builder.add_edge(START, "conversation_manager")
        builder.add_edge("conversation_manager", "context_llm")
        builder.add_conditional_edges(
            "context_llm",
            tools_condition,
            {"tools": "enhanced_tools", "__end__": END},
        )
        builder.add_edge("enhanced_tools", "context_llm")

        return builder.compile(checkpointer=self.memory)

    def _manage_conversation(self, state: ConversationState) -> ConversationState:
        messages = self.conversation_manager.trim_history(state["messages"])
        summary  = self.conversation_manager.summarize_conversation(messages)
        return {
            "messages":            messages,
            "conversation_summary": summary,
            "user_context":        state.get("user_context", {}),
            "tool_results_cache":  state.get("tool_results_cache", {}),
            "error_count":         state.get("error_count", 0),
            "last_tool_used":      state.get("last_tool_used", ""),
            "current_model_used":  state.get("current_model_used", ""),
            "model_switch_count":  state.get("model_switch_count", 0),
        }

    def _initial_state(self, message: str) -> dict:
        return {
            "messages":            [HumanMessage(content=message)],
            "user_context":        {},
            "tool_results_cache":  {},
            "conversation_summary": "",
            "error_count":         0,
            "last_tool_used":      "",
            "current_model_used":  "",
            "model_switch_count":  0,
        }

    def chat(self, message: str, thread_id: str = "default") -> str:
        """Non-streaming invoke (sidebar example buttons)."""
        try:
            # 12 allows up to 5 tool calls (1 conv_manager + 2×5 tool cycles + 1 final answer)
            # before the recursion limit triggers. Claude Opus is more thorough than Groq
            # and may call 2-3 tools per query, so 5 was too tight.
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
            result = self.graph.invoke(self._initial_state(message), config)
            ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
            return ai_msgs[-1].content if ai_msgs else "I couldn't generate a response. Please try again."
        except Exception as e:
            return f"Error: {e}. Please check your API keys and try again."

    def stream_response(self, message: str, thread_id: str = "default"):
        """
        Yields string chunks for st.write_stream (token-by-token streaming).

        - ToolMessage      → yields a brief italic progress line so the UI isn't blank
        - AIMessageChunk   → extracts text and yields it
        - Complete AIMessage → error handler response, yielded whole

        Provider differences handled here:
          - Groq / OpenAI: chunk.content is a plain string
          - Anthropic:     chunk.content is a list of blocks, e.g.
                           [{'type': 'text', 'text': 'Hello', 'index': 0}]
                           We extract only the 'text' blocks and join them.
        """
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
        st.session_state._last_tools_used = []
        try:
            for chunk, _ in self.graph.stream(
                self._initial_state(message), config, stream_mode="messages"
            ):
                if isinstance(chunk, ToolMessage):
                    name = getattr(chunk, "name", None) or "tool"
                    if name not in st.session_state._last_tools_used:
                        st.session_state._last_tools_used.append(name)
                    yield f"\n*Searching with **{name}**...*\n\n"

                elif isinstance(chunk, AIMessageChunk):
                    content = chunk.content
                    if isinstance(content, list):
                        # Anthropic format: list of typed blocks — extract text blocks only
                        text = "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    else:
                        # OpenAI / Groq format: plain string
                        text = content or ""
                    if text:
                        yield text

                elif isinstance(chunk, AIMessage) and chunk.content:
                    # All-models-failed error handler returns a complete AIMessage
                    content = chunk.content
                    if isinstance(content, list):
                        text = "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                        yield text or str(content)
                    else:
                        yield content
        except Exception as e:
            yield f"\n\nError: {e}"


# ---------------------------------------------------------------------------
# Auth page
# ---------------------------------------------------------------------------

def _show_auth_page() -> None:
    """Login / register screen shown when no active session exists."""
    # Hero
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 60%,#0f172a 100%);
                border:1px solid rgba(59,130,246,0.25);border-radius:16px;
                padding:32px 36px;margin-bottom:28px;text-align:center">
        <div style="font-size:52px;margin-bottom:12px">🔬</div>
        <h1 style="margin:0;color:#e2e8f0;font-size:28px;font-weight:700;letter-spacing:-0.5px">
            Research Intelligence Assistant
        </h1>
        <p style="margin:10px 0 0;color:#94a3b8;font-size:14px;max-width:480px;margin-left:auto;margin-right:auto">
            Multi-agent AI research platform with 13 specialized tools,
            PDF paper analysis, real-time academic search, and intelligent synthesis.
        </p>
        <div style="margin-top:16px;display:flex;justify-content:center;gap:8px;flex-wrap:wrap">
            <span style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);
                         color:#60a5fa;padding:4px 12px;border-radius:20px;font-size:12px">LangGraph</span>
            <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);
                         color:#a78bfa;padding:4px 12px;border-radius:20px;font-size:12px">Anthropic / OpenAI / Groq</span>
            <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);
                         color:#a78bfa;padding:4px 12px;border-radius:20px;font-size:12px">ChromaDB RAG</span>
            <span style="background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);
                         color:#fbbf24;padding:4px 12px;border-radius:20px;font-size:12px">13 Research Tools</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

        with tab_login:
            st.markdown("#### Welcome back")
            email    = st.text_input("Email",    key="login_email",    placeholder="you@example.com")
            password = st.text_input("Password", key="login_password", type="password")
            if st.button("Login", use_container_width=True, type="primary"):
                if not email or not password:
                    st.error("Please fill in both fields.")
                else:
                    ok, result = login_user(email, password)
                    if ok:
                        st.session_state.logged_in_user = result
                        st.rerun()
                    else:
                        st.error(result.get("error", "Login failed."))

        with tab_register:
            st.markdown("#### Create an account")
            reg_email    = st.text_input("Email",            key="reg_email",    placeholder="you@example.com")
            reg_password = st.text_input("Password",         key="reg_password", type="password", help="At least 6 characters")
            reg_confirm  = st.text_input("Confirm password", key="reg_confirm",  type="password")
            if st.button("Register", use_container_width=True, type="primary"):
                if not reg_email or not reg_password or not reg_confirm:
                    st.error("Please fill in all fields.")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                else:
                    ok, err = register_user(reg_email, reg_password)
                    if ok:
                        st.success("Account created! You can now log in.")
                    else:
                        st.error(err)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    # Auth gate
    if "logged_in_user" not in st.session_state:
        _show_auth_page()
        st.stop()

    user = st.session_state.logged_in_user

    # ── Resolve API keys ───────────────────────────────────────────────────
    # Priority: user-entered key > .env / Streamlit secrets
    user_anthropic_key = st.session_state.get("user_anthropic_key", "").strip()
    env_anthropic_key  = os.getenv("ANTHROPIC_API_KEY", "").strip()
    active_anthropic   = user_anthropic_key or env_anthropic_key

    user_openai_key = st.session_state.get("user_openai_key", "").strip()
    env_openai_key  = os.getenv("OPENAI_API_KEY", "").strip()
    active_openai   = user_openai_key or env_openai_key

    user_groq_key   = st.session_state.get("user_groq_key", "").strip()
    env_groq_key    = os.getenv("GROQ_API_KEY", "").strip()
    active_groq     = user_groq_key or env_groq_key

    # Write resolved keys back so langchain providers pick them up
    if active_anthropic:
        os.environ["ANTHROPIC_API_KEY"] = active_anthropic
    if active_openai:
        os.environ["OPENAI_API_KEY"] = active_openai
    if active_groq:
        os.environ["GROQ_API_KEY"] = active_groq

    # Anthropic takes priority — EnhancedLLM.__init__ reads env vars in this same order
    active_key = active_anthropic or active_openai or active_groq

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:

        # ── Account section ───────────────────────────────────────────────
        st.markdown("### 👤 Account")
        today_count = get_chat_count_today(user["id"])
        if user["is_admin"]:
            st.markdown(
                f"<div style='background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);"
                f"border-radius:8px;padding:10px 12px;font-size:13px'>"
                f"<b style='color:#34d399'>⚡ Admin</b><br>"
                f"<span style='color:#94a3b8'>{user['email']}</span><br>"
                f"<span style='color:#64748b;font-size:11px'>Unlimited chats</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            remaining = max(0, DAILY_LIMIT - today_count)
            bar_pct   = int(today_count / DAILY_LIMIT * 100)
            bar_color = "#22c55e" if remaining > 5 else ("#f59e0b" if remaining > 0 else "#ef4444")
            st.markdown(
                f"<div style='background:rgba(30,41,59,0.5);border:1px solid #1e293b;"
                f"border-radius:8px;padding:10px 12px;font-size:13px'>"
                f"<b style='color:#e2e8f0'>{user['email']}</b><br>"
                f"<div style='margin:6px 0 3px;background:#1e293b;border-radius:4px;height:4px'>"
                f"<div style='width:{bar_pct}%;background:{bar_color};height:4px;border-radius:4px;transition:width 0.3s'></div>"
                f"</div>"
                f"<span style='color:#64748b;font-size:11px'>{today_count}/{DAILY_LIMIT} chats today · {remaining} remaining</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if st.button("🚪 Logout", use_container_width=True):
            for k in ("logged_in_user", "chatbot_initialized", "chatbot", "messages", "thread_id",
                      "user_anthropic_key", "user_openai_key", "user_groq_key"):
                st.session_state.pop(k, None)
            st.rerun()

        st.markdown("---")

        # ── AI Provider section ───────────────────────────────────────────
        st.markdown("### 📡 AI Provider")

        # Show which provider is currently active
        if active_anthropic:
            provider_label  = "Claude Opus 4.8 (Anthropic)"
            provider_color  = "#a78bfa"
            provider_bg     = "rgba(124,58,237,0.1)"
            provider_border = "rgba(124,58,237,0.3)"
        elif active_openai:
            provider_label  = "GPT-4o (OpenAI)"
            provider_color  = "#34d399"
            provider_bg     = "rgba(16,185,129,0.1)"
            provider_border = "rgba(16,185,129,0.3)"
        elif active_groq:
            provider_label  = "Llama 3.3 (Groq)"
            provider_color  = "#fbbf24"
            provider_bg     = "rgba(245,158,11,0.1)"
            provider_border = "rgba(245,158,11,0.3)"
        else:
            provider_label  = "No API key configured"
            provider_color  = "#ef4444"
            provider_bg     = "rgba(239,68,68,0.1)"
            provider_border = "rgba(239,68,68,0.3)"

        st.markdown(
            f"<div style='background:{provider_bg};border:1px solid {provider_border};"
            f"border-radius:8px;padding:8px 12px;margin-bottom:8px'>"
            f"<span style='width:8px;height:8px;border-radius:50%;background:{provider_color};"
            f"display:inline-block;margin-right:8px'></span>"
            f"<span style='color:{provider_color};font-size:13px;font-weight:600'>{provider_label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Dropdown to choose which provider key to configure
        key_provider = st.selectbox(
            "Configure API key for:",
            [
                "🤖 Anthropic (Claude Opus — Recommended)",
                "🔑 OpenAI (GPT-4o)",
                "⚡ Groq (Free tier / Fallback)",
            ],
            index=0,
            label_visibility="visible",
            key="key_provider_select",
        )

        if "Anthropic" in key_provider:
            st.caption("Get a key at [console.anthropic.com](https://console.anthropic.com/)")
            new_anthropic = st.text_input(
                "Anthropic key", type="password", placeholder="sk-ant-...",
                value=st.session_state.get("user_anthropic_key", ""),
                label_visibility="collapsed", key="anthropic_key_input",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Apply", key="apply_anthropic", use_container_width=True):
                    if new_anthropic.strip() != st.session_state.get("user_anthropic_key", ""):
                        st.session_state.user_anthropic_key = new_anthropic.strip()
                        st.session_state.pop("chatbot_initialized", None)
                        st.session_state.pop("chatbot", None)
                    st.rerun()
            with c2:
                if st.button("Clear", key="remove_anthropic", use_container_width=True):
                    st.session_state.user_anthropic_key = ""
                    st.session_state.pop("chatbot_initialized", None)
                    st.session_state.pop("chatbot", None)
                    st.rerun()

        elif "OpenAI" in key_provider:
            st.caption("Get a key at [platform.openai.com](https://platform.openai.com/)")
            new_openai = st.text_input(
                "OpenAI key", type="password", placeholder="sk-proj-...",
                value=st.session_state.get("user_openai_key", ""),
                label_visibility="collapsed", key="openai_key_input",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Apply", key="apply_openai", use_container_width=True):
                    if new_openai.strip() != st.session_state.get("user_openai_key", ""):
                        st.session_state.user_openai_key = new_openai.strip()
                        st.session_state.pop("chatbot_initialized", None)
                        st.session_state.pop("chatbot", None)
                    st.rerun()
            with c2:
                if st.button("Clear", key="remove_openai", use_container_width=True):
                    st.session_state.user_openai_key = ""
                    st.session_state.pop("chatbot_initialized", None)
                    st.session_state.pop("chatbot", None)
                    st.rerun()

        else:  # Groq
            st.caption("Free at [console.groq.com](https://console.groq.com/)")
            new_groq = st.text_input(
                "Groq key", type="password", placeholder="gsk_...",
                value=st.session_state.get("user_groq_key", ""),
                label_visibility="collapsed", key="groq_key_input",
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Apply", key="apply_groq", use_container_width=True):
                    if new_groq.strip() != st.session_state.get("user_groq_key", ""):
                        st.session_state.user_groq_key = new_groq.strip()
                        st.session_state.pop("chatbot_initialized", None)
                        st.session_state.pop("chatbot", None)
                    st.rerun()
            with c2:
                if st.button("Clear", key="remove_groq", use_container_width=True):
                    st.session_state.user_groq_key = ""
                    st.session_state.pop("chatbot_initialized", None)
                    st.session_state.pop("chatbot", None)
                    st.rerun()

        st.markdown("---")

        # ── Loaded Papers section (only shown when papers are loaded) ────────
        loaded = st.session_state.get("uploaded_pdfs", {})
        if loaded:
            st.markdown("---")
            st.markdown("### 📄 Loaded Papers")
            for pdf_name, pdf_info in list(loaded.items()):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"**{pdf_name}**")
                    if isinstance(pdf_info, dict):
                        st.caption(f"{pdf_info.get('pages','?')} pages · {pdf_info.get('chunks','?')} chunks")
                with c2:
                    if st.button("✕", key=f"rm_{pdf_name}", help=f"Remove {pdf_name}"):
                        try:
                            from src.rag import ResearchVectorStore
                            ResearchVectorStore().delete_pdf(pdf_name, str(user["id"]))
                        except Exception:
                            pass
                        del st.session_state.uploaded_pdfs[pdf_name]
                        st.rerun()

        st.markdown("---")

        # ── Tools section ─────────────────────────────────────────────────
        with st.expander("🛠️ Research Tools (13)", expanded=False):
            st.markdown("""
**Academic Search**
- ArXiv · PubMed · Semantic Scholar
- OpenAlex · Find Related Papers

**Web & General**
- DuckDuckGo · Tavily · Wikipedia

**Analysis & Utilities**
- Calculator · Code Analyzer
- Weather · File Generator
- PDF Search (uploaded papers)
""")

        st.markdown("---")

        # ── Settings ──────────────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared.")
            st.rerun()

        # ── Example Queries ───────────────────────────────────────────────
        with st.expander("💡 Example Queries", expanded=False):
            examples = [
                "Find the latest papers on large language models",
                "Search PubMed: recent CRISPR gene editing studies",
                "Find highly-cited transformer architecture papers",
                "Calculate compound interest: $5000 at 7% for 10 years",
                "Explain attention mechanism with code and diagrams",
                "What are the current debates in quantum computing research?",
            ]
            for i, ex in enumerate(examples):
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    st.session_state.pending_example = ex
                    st.rerun()

        st.markdown("---")

        # ── About / Portfolio ─────────────────────────────────────────────
        st.markdown("### 🔗 About")
        st.markdown("""
<div style="font-size:12px;color:#64748b;line-height:1.7">
  <b style="color:#94a3b8">Stack:</b> Streamlit · LangGraph · LangChain<br>
  <b style="color:#94a3b8">LLMs:</b> Claude Opus 4.8 · GPT-4o · Groq Llama 3.3<br>
  <b style="color:#94a3b8">RAG:</b> ChromaDB · sentence-transformers<br>
  <b style="color:#94a3b8">Auth:</b> SQLite · bcrypt<br>
  <b style="color:#94a3b8">Search:</b> arXiv · PubMed · Semantic Scholar
</div>
""", unsafe_allow_html=True)
        st.markdown("[GitHub](https://github.com/Waseekk/multi-tool-research-bot) · [Report Issue](https://github.com/Waseekk/multi-tool-research-bot/issues)")

    # ── Hero header ────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#0f172a 100%);
            border:1px solid rgba(59,130,246,0.2);border-radius:14px;
            padding:20px 26px;margin-bottom:18px">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
    <span style="font-size:36px">🔬</span>
    <div style="flex:1;min-width:200px">
      <h1 style="margin:0;color:#e2e8f0;font-size:22px;font-weight:700;letter-spacing:-0.3px">
        Research Intelligence Assistant
      </h1>
      <p style="margin:4px 0 0;color:#94a3b8;font-size:12.5px">
        Multi-agent AI &nbsp;·&nbsp; 13 research tools &nbsp;·&nbsp; PDF analysis &nbsp;·&nbsp; Real-time synthesis
      </p>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap">
      <span style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.3);
                   color:#60a5fa;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500">LangGraph</span>
      <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);
                   color:#a78bfa;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500">Claude / GPT-4o / Groq</span>
      <span style="background:rgba(124,58,237,0.15);border:1px solid rgba(124,58,237,0.3);
                   color:#a78bfa;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:500">ChromaDB</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Gate: require at least one API key ─────────────────────────────────
    if not active_key:
        st.warning("⚠️ No API key found. Enter your key in the sidebar to start.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**Anthropic** (Recommended)  \nClaude Opus 4.8 at [console.anthropic.com](https://console.anthropic.com/)")
        with c2:
            st.info("**OpenAI** (GPT-4o)  \nGet a key at [platform.openai.com](https://platform.openai.com/)")
        with c3:
            st.info("**Groq** (Free tier)  \nGet a key at [console.groq.com](https://console.groq.com/)")
        st.stop()

    # ── Session init ────────────────────────────────────────────────────────
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── PDF upload panel ────────────────────────────────────────────────────
    with st.expander(
        "📄 Upload Research Papers",
        expanded=not bool(st.session_state.get("uploaded_pdfs")),
    ):
        if not st.session_state.get("_rag_warned"):
            st.info("First upload downloads the embedding model (~420 MB, one-time). Subsequent uploads are fast.")
            st.session_state._rag_warned = True
        files = st.file_uploader(
            "Drop PDF files here or click to browse",
            type=["pdf"],
            accept_multiple_files=True,
            key="main_pdf_uploader",
        )
        if files:
            if "uploaded_pdfs" not in st.session_state:
                st.session_state.uploaded_pdfs = {}
            for f in files:
                if f.name not in st.session_state.uploaded_pdfs:
                    with st.spinner(f"Indexing {f.name}..."):
                        try:
                            from src.rag import PDFProcessor, ResearchVectorStore
                            pdf_data = PDFProcessor().process_pdf(f.read(), f.name)
                            n_chunks = ResearchVectorStore().add_pdf(pdf_data, str(user["id"]))
                            st.session_state.uploaded_pdfs[f.name] = {
                                "pages":  pdf_data["metadata"]["pages"],
                                "chunks": n_chunks,
                                "title":  pdf_data["metadata"]["title"],
                            }
                            st.success(
                                f"✅ **{f.name}** — "
                                f"{pdf_data['metadata']['pages']} pages · {n_chunks} chunks indexed"
                            )
                        except Exception as e:
                            st.error(f"Failed to process {f.name}: {e}")
        if st.session_state.get("uploaded_pdfs"):
            st.markdown(
                "**Loaded:** " + " &nbsp;·&nbsp; ".join(
                    f"📄 {n}" for n in st.session_state.uploaded_pdfs
                )
            )

    # ── Chat history ────────────────────────────────────────────────────────
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                tools_used = msg.get("tools_used", [])
                if tools_used:
                    st.markdown(render_tool_badges(tools_used), unsafe_allow_html=True)
                render_copy_btn(msg["content"], f"hist_{idx}")

    # ── Chat input ──────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything — academic papers, PDFs, calculations, code, and more…")
    if "pending_example" in st.session_state:
        user_input = st.session_state.pop("pending_example")

    def _is_inappropriate(text: str) -> bool:
        blocked = ["fuck", "shit", "bitch", "asshole", "bastard", "dick", "pussy", "cunt"]
        t = text.lower()
        return any(w in t for w in blocked)

    if user_input:
        if _is_inappropriate(user_input):
            with st.chat_message("assistant"):
                st.warning("This assistant is designed for research. Please keep queries respectful.")
            st.stop()

        if is_daily_limit_reached(user["id"], user["is_admin"]):
            st.warning(
                f"You've reached today's limit of {DAILY_LIMIT} chats. "
                "Come back tomorrow — your limit resets at midnight."
            )
            st.stop()

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                chatbot = StreamlitChatbot()
                # Set context vars before graph.stream() — tools read these instead of
                # st.session_state (which is not reliably accessible in LangGraph threads)
                _active_user_id.set(str(user["id"]))
                _active_pdf_names.set(list(st.session_state.get("uploaded_pdfs", {}).keys()))

                response = st.write_stream(
                    chatbot.stream_response(user_input, thread_id=st.session_state.thread_id)
                )
                # st.write_stream may return a StreamingOutput object (not str) in some
                # Streamlit builds — especially when the stream ends with an error chunk.
                # Cast to str so render_copy_btn and session_state storage are always safe.
                response = str(response) if not isinstance(response, str) else response
                increment_chat_count(user["id"])

                used_tools = st.session_state.get("_last_tools_used", [])
                if used_tools:
                    st.markdown(render_tool_badges(used_tools), unsafe_allow_html=True)

                # Small copy icon — replaces the old "📋 Copy response" expander
                render_copy_btn(response, f"new_{len(st.session_state.messages)}")

                st.session_state.messages.append({
                    "role":       "assistant",
                    "content":    response,
                    "tools_used": used_tools,
                })
            except Exception as e:
                err = f"Sorry, I encountered an error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})


if __name__ == "__main__":
    main()
