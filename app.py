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

Modules this file depends on:
  src/tools.py       — initialize_tools()
  src/models.py      — EnhancedLLM, ConversationState
  src/nodes.py       — create_enhanced_nodes()
  src/conversation.py — ConversationManager
"""

import streamlit as st
import os
from typing import List, Dict, Any
import json
from datetime import datetime

# Handles local .env file; environment variables set here are visible to all modules
from dotenv import load_dotenv
load_dotenv()

# For Streamlit Cloud: Load secrets if available
if hasattr(st, 'secrets'):
    try:
        # Set environment variables from Streamlit secrets
        if 'GROQ_API_KEY' in st.secrets:
            os.environ['GROQ_API_KEY'] = st.secrets['GROQ_API_KEY']
        if 'TAVILY_API_KEY' in st.secrets:
            os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']
    except Exception as e:
        pass  # Secrets not configured yet

# LangChain and LangGraph imports
import uuid
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, AIMessageChunk, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import our custom modules (from src package)
from src.tools import initialize_tools
from src.models import EnhancedLLM, ConversationState
from src.nodes import create_enhanced_nodes
from src.conversation import ConversationManager

# Page config
st.set_page_config(
    page_title="Multi-Tool Research Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .main > div {
        padding-top: 2rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitChatbot:
    """
    Wraps the LangGraph compiled graph and exposes chat/stream interfaces to the UI.

    Streamlit reruns the entire script on every user interaction, so the chatbot
    must be stored in `st.session_state` to survive across reruns. On the first
    load we initialize everything and cache it; on subsequent reruns we restore
    from session state rather than rebuilding the graph and reloading tools.
    """

    def __init__(self):
        if "chatbot_initialized" not in st.session_state:
            with st.spinner("Initializing chatbot and loading tools..."):
                self.tools = initialize_tools()
                self.llm_manager = EnhancedLLM()
                self.conversation_manager = ConversationManager()
                # MemorySaver stores conversation checkpoints in memory (no DB needed).
                # Each thread_id gets its own isolated message history.
                self.memory = MemorySaver()
                self.graph = self._build_graph()
                st.session_state.chatbot_initialized = True
                st.session_state.chatbot = self
        else:
            chatbot = st.session_state.chatbot
            self.tools = chatbot.tools
            self.llm_manager = chatbot.llm_manager
            self.conversation_manager = chatbot.conversation_manager
            self.memory = chatbot.memory
            self.graph = chatbot.graph
    
    def _build_graph(self) -> StateGraph:
        """
        Assembles the LangGraph ReAct loop:

          conversation_manager -> context_llm -> (tools? -> context_llm -> ...) -> END

        tools_condition is LangGraph's built-in router: it reads the last AI message
        and routes to "enhanced_tools" if it contains tool call requests, or to END
        if it's a plain text response. This loop continues until the LLM stops
        requesting tools, at which point it streams the final answer.
        """
        context_llm, enhanced_tools = create_enhanced_nodes(self.tools, self.llm_manager)

        builder = StateGraph(ConversationState)
        builder.add_node("conversation_manager", self._manage_conversation)
        builder.add_node("context_llm", context_llm)
        builder.add_node("enhanced_tools", enhanced_tools)

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
        """Manage conversation context with enhanced tracking"""
        messages = self.conversation_manager.trim_history(state["messages"])
        summary = self.conversation_manager.summarize_conversation(messages)
        
        return {
            "messages": messages,
            "conversation_summary": summary,
            "user_context": state.get("user_context", {}),
            "tool_results_cache": state.get("tool_results_cache", {}),
            "error_count": state.get("error_count", 0),
            "last_tool_used": state.get("last_tool_used", ""),
            "current_model_used": state.get("current_model_used", ""),
            "model_switch_count": state.get("model_switch_count", 0)
        }
    
    def _initial_state(self, message: str) -> dict:
        return {
            "messages": [HumanMessage(content=message)],
            "user_context": {},
            "tool_results_cache": {},
            "conversation_summary": "",
            "error_count": 0,
            "last_tool_used": "",
            "current_model_used": "",
            "model_switch_count": 0,
        }

    def chat(self, message: str, thread_id: str = "default") -> str:
        """Non-streaming chat — used for sidebar example buttons"""
        try:
            # recursion_limit prevents infinite tool-calling loops where a weak
            # fallback model keeps re-calling tools instead of writing a final answer
            config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
            result = self.graph.invoke(self._initial_state(message), config)
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            return ai_messages[-1].content if ai_messages else "I couldn't generate a response. Please try again."
        except Exception as e:
            return f"Error: {str(e)}. Please check your API keys and try again."

    def stream_response(self, message: str, thread_id: str = "default"):
        """
        Yields string chunks suitable for st.write_stream (token-by-token streaming).

        Two types of chunks are emitted:
        1. Tool indicator — ToolMessage signals a tool just ran; we yield a brief italic
           line so the user sees progress instead of a blank screen during multi-second
           API calls (e.g. ArXiv searches).
        2. LLM tokens — AIMessageChunk with text content. We skip chunks that only carry
           tool_call_chunks (the LLM choosing which tool to call) and surface only the
           final human-readable answer tokens.

        NOTE: We intentionally do NOT filter by langgraph_node name. The metadata key
        "langgraph_node" changed or is absent in some LangGraph versions, which caused
        every chunk to be silently dropped on tool-using queries. Filtering by message
        type alone is sufficient and more robust across versions:
          - ToolMessage       → only ever comes from the tool execution node
          - AIMessageChunk    → only streamed by LLM nodes (conversation_manager and
                                enhanced_tools never call an LLM)
          - AIMessage (whole) → returned by the error handler when all models fail
        """
        # recursion_limit prevents infinite tool-calling loops where a weak
        # fallback model keeps re-calling tools instead of writing a final answer
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
        try:
            for chunk, metadata in self.graph.stream(
                self._initial_state(message), config, stream_mode="messages"
            ):
                if isinstance(chunk, ToolMessage):
                    # Show a progress indicator while the tool runs so the UI isn't blank
                    yield f"\n*Searching with {chunk.name}...*\n\n"
                elif (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    # tool_call_chunks = LLM deciding which tool to call; skip those
                    and not getattr(chunk, "tool_call_chunks", None)
                ):
                    yield chunk.content
                elif isinstance(chunk, AIMessage) and chunk.content:
                    # The all-models-failed error handler returns a complete AIMessage,
                    # not streamed chunks — catch it here so it isn't silently dropped
                    yield chunk.content
        except Exception as e:
            yield f"\n\nError: {str(e)}"

def main():
    """Main Streamlit application"""

    # --- Resolve active API key ---
    # Priority: key entered in the sidebar > key from .env / HuggingFace Secrets
    user_key = st.session_state.get("user_groq_key", "").strip()
    env_key  = os.getenv("GROQ_API_KEY", "").strip()
    active_key = user_key or env_key
    # Write the resolved key back to the environment so ChatGroq picks it up
    if active_key:
        os.environ["GROQ_API_KEY"] = active_key
    
    # Title and description
    st.title("🤖 Multi-Tool Research Bot")
    st.markdown("""
    **An intelligent AI assistant with multiple research tools**

    This bot can help you with:
    - 📚 Academic research (ArXiv, PubMed, Semantic Scholar, OpenAlex)
    - 🏥 Medical & biomedical research (PubMed)
    - 🌐 Web search and current information
    - 📖 Wikipedia knowledge base
    - 🧮 Mathematical calculations
    - 💻 Code analysis and generation
    - 🌤️ Weather information
    - 📄 File content generation
    """)
    
    # Sidebar
    with st.sidebar:
        # --- API key input ---
        st.header("🔑 Groq API Key")
        with st.expander("Use your own key", expanded=not active_key):
            st.markdown(
                "Get a **free** key at [console.groq.com](https://console.groq.com/). "
                "Your key is only stored in your browser session — never saved anywhere."
            )
            entered_key = st.text_input(
                "Paste key here",
                type="password",
                placeholder="gsk_...",
                value=st.session_state.get("user_groq_key", ""),
                label_visibility="collapsed",
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Apply", use_container_width=True):
                    new_key = entered_key.strip()
                    if new_key != st.session_state.get("user_groq_key", ""):
                        st.session_state.user_groq_key = new_key
                        # Force chatbot to reinitialize with the new key
                        st.session_state.pop("chatbot_initialized", None)
                        st.session_state.pop("chatbot", None)
                    st.rerun()
            with col2:
                if st.button("Remove", use_container_width=True):
                    st.session_state.user_groq_key = ""
                    st.session_state.pop("chatbot_initialized", None)
                    st.session_state.pop("chatbot", None)
                    st.rerun()

        st.markdown("---")
        st.header("🛠️ Available Tools")
        st.markdown("""
        **Research Sources:**
        - **ArXiv**: Physics, Math, CS papers
        - **PubMed**: Medical/biomedical research
        - **Semantic Scholar**: Highly-cited papers
        - **OpenAlex**: Open access works
        - **Wikipedia**: General knowledge
        - **Web Search**: Current information

        **Utilities:**
        - **Calculator**: Math operations
        - **Code Analyzer**: Code review
        - **Weather Info**: Location weather
        - **File Generator**: Sample files
        """)
        
        st.header("⚙️ Settings")
        if st.button("🗑️ Clear Chat History"):
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
        
        st.header("📝 Example Queries")
        examples = [
            "Find the latest ArXiv papers on large language models",
            "Search PubMed for recent studies on CRISPR gene editing",
            "Find highly cited papers on transformer architecture using Semantic Scholar",
            "Calculate compound interest: 5000 at 7% annual rate for 10 years",
            "Search recent news about AI regulation",
            "Analyze this Python code: def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)",
        ]
        
        for i, example in enumerate(examples):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                # Stage the example as a pending input; the main chat loop
                # handles it with the same streaming path as regular messages.
                st.session_state.pending_example = example
                st.rerun()
        
        # Add info about deployment
        st.markdown("---")
        st.markdown("### 📊 App Info")
        if active_key:
            key_source = "your key" if user_key else "shared key"
            st.info(f"**Status:** ✅ Running\n\n**Key:** {key_source}")
        else:
            st.error("**Status:** ❌ No API key")
        
        # GitHub link
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[View on GitHub](https://github.com/Waseekk/multi-tool-research-bot)")
    
    # Gate: require a key before the chatbot initializes.
    # Placed here (after sidebar) so the key input is always visible even with no key.
    if not active_key:
        st.warning("⚠️ No Groq API key found. Enter your key in the sidebar to get started.")
        st.info("Get a **free** key in 30 seconds at [console.groq.com](https://console.groq.com/) — no credit card needed.")
        st.stop()

    # Each browser session gets a unique thread_id. MemorySaver uses this as the
    # key to isolate conversation history — without it all users would share one
    # memory and see each other's messages.
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input — also picks up example button clicks staged via pending_example
    user_input = st.chat_input("Ask me anything! I have access to multiple research tools.")
    if "pending_example" in st.session_state:
        user_input = st.session_state.pop("pending_example")

    def _is_inappropriate(text: str) -> bool:
        blocked = ["fuck", "shit", "bitch", "asshole", "bastard", "dick", "pussy", "cunt", "nigger", "faggot"]
        t = text.lower()
        return any(w in t for w in blocked)

    if user_input:
        if _is_inappropriate(user_input):
            with st.chat_message("assistant"):
                st.warning("This assistant is designed for research and professional use. Please keep queries respectful.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                chatbot = StreamlitChatbot()
                response = st.write_stream(
                    chatbot.stream_response(user_input, thread_id=st.session_state.thread_id)
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, LangChain, and LangGraph | [Report Issues](https://github.com/Waseekk/multi-tool-research-bot/issues)")

if __name__ == "__main__":
    main()