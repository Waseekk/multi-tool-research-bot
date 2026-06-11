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
            config = {"configurable": {"thread_id": thread_id}}
            result = self.graph.invoke(self._initial_state(message), config)
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            return ai_messages[-1].content if ai_messages else "I couldn't generate a response. Please try again."
        except Exception as e:
            return f"Error: {str(e)}. Please check your API keys and try again."

    def stream_response(self, message: str, thread_id: str = "default"):
        """
        Yields string chunks suitable for st.write_stream (token-by-token streaming).

        Two types of chunks are emitted:
        1. Tool indicator — when a ToolMessage arrives from enhanced_tools, we yield
           a brief italic line so the user sees progress instead of a blank screen
           during multi-second API calls (e.g. ArXiv searches).
        2. LLM tokens — AIMessageChunk.content from context_llm. We filter on
           `langgraph_node == "context_llm"` because other nodes (conversation_manager,
           enhanced_tools) also emit messages we don't want to surface raw. We also
           skip chunks that only contain tool_call_chunks — those are the LLM
           deciding which tool to call, not text meant for the user.
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            for chunk, metadata in self.graph.stream(
                self._initial_state(message), config, stream_mode="messages"
            ):
                node = metadata.get("langgraph_node", "")
                if node == "enhanced_tools" and isinstance(chunk, ToolMessage):
                    yield f"\n*Searching with {chunk.name}...*\n\n"
                elif (
                    node == "context_llm"
                    and isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    and not getattr(chunk, "tool_call_chunks", None)
                ):
                    yield chunk.content
        except Exception as e:
            yield f"\n\nError: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Check for API key FIRST
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.error("⚠️ GROQ_API_KEY not found!")
        st.info("""
        **To fix this:**
        
        **For Streamlit Cloud:**
        1. Go to your app settings
        2. Click on "Secrets" 
        3. Add: `GROQ_API_KEY = "your_key_here"`
        
        **For Local Development:**
        1. Create a `.env` file
        2. Add: `GROQ_API_KEY=your_key_here`
        
        **Get your API key from:** https://console.groq.com/
        """)
        st.stop()
    
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
                # Initialize messages if not exists
                if 'messages' not in st.session_state:
                    st.session_state.messages = []
                    
                st.session_state.messages.append({"role": "user", "content": example})
                
                with st.spinner("Processing your query..."):
                    try:
                        chatbot = StreamlitChatbot()
                        thread_id = st.session_state.get("thread_id", str(uuid.uuid4()))
                        response = chatbot.chat(example, thread_id=thread_id)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.rerun()
        
        # Add info about deployment
        st.markdown("---")
        st.markdown("### 📊 App Info")
        st.info(f"**Status:** {'✅ Running' if groq_key else '❌ No API Key'}")
        
        # GitHub link
        st.markdown("---")
        st.markdown("### 🔗 Links")
        st.markdown("[📂 View on GitHub](https://github.com/YOUR_USERNAME/multi-tool-research-bot)")
    
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
    
    # Chat input
    user_input = st.chat_input("Ask me anything! I have access to multiple research tools.")
    
    if user_input:
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
    st.markdown("Built with ❤️ using Streamlit, LangChain, and LangGraph | [Report Issues](https://github.com/YOUR_USERNAME/multi-tool-research-bot/issues)")

if __name__ == "__main__":
    main()