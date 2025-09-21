import streamlit as st
import os
from typing import List, Dict, Any
import json
from datetime import datetime

# Environment setup - MUST be first!
from dotenv import load_dotenv
load_dotenv()

# LangChain and LangGraph imports
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
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

class StreamlitChatbot:
    """Streamlit-optimized chatbot"""
    
    def __init__(self):
        if 'chatbot_initialized' not in st.session_state:
            self.tools = initialize_tools()
            self.llm_manager = EnhancedLLM()
            self.conversation_manager = ConversationManager()
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
        """Build the conversation graph with proper tool execution flow"""
        context_llm, enhanced_tools = create_enhanced_nodes(self.tools, self.llm_manager)
        
        builder = StateGraph(ConversationState)
        
        # Add nodes
        builder.add_node("conversation_manager", self._manage_conversation)
        builder.add_node("context_llm", context_llm)
        builder.add_node("enhanced_tools", enhanced_tools)
        
        # Define the flow
        builder.add_edge(START, "conversation_manager")
        builder.add_edge("conversation_manager", "context_llm")
        
        # CRITICAL: This determines if tools should be called or if we're done
        builder.add_conditional_edges(
            "context_llm",
            tools_condition,  # This checks if the LLM response contains tool calls
            {
                "tools": "enhanced_tools",  # If tool calls found, go to tools
                "__end__": END              # If no tool calls, end conversation
            }
        )
        
        # After tools execute, go back to LLM to process results
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
    
    def chat(self, message: str, thread_id: str = "streamlit") -> str:
        """Chat interface for Streamlit"""
        try:
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_context": {},
                "tool_results_cache": {},
                "conversation_summary": "",
                "error_count": 0,
                "last_tool_used": ""
            }
            
            config = {"configurable": {"thread_id": thread_id}}
            result = self.graph.invoke(initial_state, config)
            
            final_messages = result["messages"]
            ai_messages = [msg for msg in final_messages if isinstance(msg, AIMessage)]
            
            if ai_messages:
                return ai_messages[-1].content
            else:
                return "I'm sorry, I couldn't generate a response. Please try again."
                
        except Exception as e:
            return f"Error: {str(e)}. Please check your API keys and try again."

def main():
    """Main Streamlit application"""
    
    # Check for API key FIRST
    if not os.getenv("GROQ_API_KEY"):
        st.error("⚠️ GROQ_API_KEY not found! Please add it to your .env file.")
        st.info("Get your free API key from: https://console.groq.com/")
        st.stop()
    
    # Title and description
    st.title("🤖 Multi-Tool Research Bot")
    st.markdown("""
    **An intelligent AI assistant with multiple research tools**
    
    This bot can help you with:
    - 📚 Academic research (ArXiv papers)
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
        - **ArXiv Search**: Research papers
        - **Wikipedia**: General knowledge
        - **Web Search**: Current information
        - **Calculator**: Math operations
        - **Code Analyzer**: Code review
        - **Weather Info**: Location weather
        - **File Generator**: Sample files
        """)
        
        st.header("⚙️ Settings")
        if st.button("Clear Chat History"):
            if 'messages' in st.session_state:
                del st.session_state.messages
            st.success("Chat history cleared!")
        
        st.header("📝 Example Queries")
        examples = [
            "Calculate 15% of 2,500",
            "Latest research on quantum computing",
            "Weather in New York",
            "Analyze this Python code: def hello(): print('Hi')",
            "Generate a CSV for student data",
            "What is machine learning?"
        ]
        
        for i, example in enumerate(examples):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": example})
                
                with st.spinner("Processing your query..."):
                    try:
                        chatbot = StreamlitChatbot()
                        response = chatbot.chat(example)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        st.rerun()
    
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
            with st.spinner("Thinking and using tools..."):
                try:
                    chatbot = StreamlitChatbot()
                    response = chatbot.chat(user_input)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, LangChain, and LangGraph")

if __name__ == "__main__":
    main()