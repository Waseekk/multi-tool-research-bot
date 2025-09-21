from typing import List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from .models import ConversationState, EnhancedLLM

def create_enhanced_nodes(tools: List, llm_manager: EnhancedLLM):
    """Create enhanced nodes with proper tool execution and error handling"""
    
    def context_aware_llm(state: ConversationState) -> ConversationState:
        """Enhanced LLM node with context awareness and proper tool binding"""
        try:
            # Get current LLM with task optimization
            messages = state["messages"]
            
            # Determine task type from the last human message
            last_human_msg = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_human_msg = msg.content
                    break
            
            # Get optimized LLM for the task
            if last_human_msg:
                llm = get_task_optimized_llm(llm_manager, last_human_msg)
            else:
                llm = llm_manager.get_llm()
            
            # IMPORTANT: Bind tools to LLM properly
            llm_with_tools = llm.bind_tools(tools)
            
            # Add system message with context if this is the first message
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                system_prompt = f"""You are an advanced AI assistant with access to multiple research and utility tools. 

Current conversation summary: {state.get('conversation_summary', 'New conversation')}
Last successful tool: {state.get('last_tool_used', 'None')}

Available tools:
- arxiv_query_run: Search academic papers on ArXiv
- wikipedia_query_run: Search Wikipedia for general knowledge
- tavily_search_results or duckduckgo_search: Web search for current information
- calculator: Perform mathematical calculations
- code_analyzer: Analyze and review code
- weather_info: Get weather information
- file_content_generator: Generate sample files

Guidelines:
1. ALWAYS use the appropriate tool when the user's question requires:
   - Current information (use web search)
   - Academic research (use ArXiv)
   - Calculations (use calculator)
   - Code analysis (use code_analyzer)
   - Weather (use weather_info)
   - File generation (use file_content_generator)

2. For the query "Latest research on quantum computing" -> use arxiv_query_run
3. For "Calculate 15% of 2,500" -> use calculator tool
4. Be concise but comprehensive in your responses
5. If a tool fails, try alternative approaches
6. Use tools proactively - don't just provide general knowledge when specific tools can give better answers

IMPORTANT: When you need to use a tool, make sure to call it properly. The system will execute the tool and provide results."""
                
                messages = [SystemMessage(content=system_prompt)] + messages
            
            # Invoke LLM with tools
            response = llm_with_tools.invoke(messages)
            
            # Log the current model being used
            current_model = llm_manager.get_current_model_name()
            print(f"Using model: {current_model}")
            
            # Update state with model information
            new_state = {
                "messages": [response],
                "error_count": 0,  # Reset error count on success
                "current_model_used": current_model
            }
            
            return new_state
            
        except Exception as e:
            error_msg = f"Error in LLM processing: {str(e)}"
            print(error_msg)
            
            # Increment error count
            error_count = state.get("error_count", 0) + 1
            
            # Try switching to secondary model if primary fails
            if error_count == 1:
                try:
                    llm = llm_manager.get_secondary_llm()
                    llm_with_tools = llm.bind_tools(tools)
                    response = llm_with_tools.invoke(messages)
                    
                    return {
                        "messages": [response],
                        "error_count": 0,
                        "current_model_used": llm_manager.get_current_model_name()
                    }
                except Exception as e2:
                    print(f"Secondary model also failed: {str(e2)}")
            
            # Fallback response
            fallback_response = AIMessage(
                content=f"I encountered an issue processing your request (attempt {error_count}). "
                       f"Let me try a different approach. Error: {str(e)}"
            )
            
            return {
                "messages": [fallback_response],
                "error_count": error_count
            }
    
    def enhanced_tool_node(state: ConversationState) -> ConversationState:
        """Enhanced tool node with proper error handling and result processing"""
        try:
            # Create tool node
            tool_node = ToolNode(tools)
            
            # Execute tools
            result = tool_node.invoke(state)
            
            # Process the result
            messages = result.get("messages", [])
            
            # Update tool cache and tracking
            if messages:
                last_message = messages[-1]
                tool_name = getattr(last_message, 'name', 'unknown_tool')
                
                # Update state with successful tool usage
                updated_state = {
                    "messages": messages,
                    "last_tool_used": tool_name,
                    "error_count": 0  # Reset error count on successful tool use
                }
                
                # Cache tool results if it's a ToolMessage
                if isinstance(last_message, ToolMessage):
                    cache_key = f"{tool_name}_{hash(str(last_message.content)[:100])}"
                    if "tool_results_cache" not in result:
                        result["tool_results_cache"] = state.get("tool_results_cache", {})
                    result["tool_results_cache"][cache_key] = {
                        "content": last_message.content,
                        "timestamp": state.get("conversation_summary", ""),
                        "tool": tool_name
                    }
                    updated_state["tool_results_cache"] = result["tool_results_cache"]
                
                result.update(updated_state)
            
            return result
            
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            print(error_msg)
            
            # Create error response
            error_response = AIMessage(
                content=f"I encountered an error while using the tools: {str(e)}. "
                       "Let me try to help you in a different way or please rephrase your question."
            )
            
            return {
                "messages": [error_response],
                "error_count": state.get("error_count", 0) + 1
            }
    
    return context_aware_llm, enhanced_tool_node

def get_task_optimized_llm(llm_manager: EnhancedLLM, user_message: str):
    """Analyze user message and return optimized LLM for the task"""
    message_lower = user_message.lower()
    
    # Task detection patterns
    if any(word in message_lower for word in ['calculate', 'math', 'equation', 'solve', 'compute', '%', 'percent']):
        return llm_manager.get_model_for_task("math")
    elif any(word in message_lower for word in ['research', 'paper', 'study', 'academic', 'arxiv', 'latest research']):
        return llm_manager.get_model_for_task("reasoning")  # Use reasoning model for research
    elif any(word in message_lower for word in ['analyze', 'analysis', 'compare', 'evaluate']):
        return llm_manager.get_model_for_task("analysis")
    elif any(word in message_lower for word in ['code', 'programming', 'python', 'javascript', 'debug']):
        return llm_manager.get_model_for_task("coding")
    elif any(word in message_lower for word in ['write', 'story', 'poem', 'creative', 'imagine']):
        return llm_manager.get_model_for_task("creative")
    elif any(word in message_lower for word in ['weather', 'temperature', 'climate']):
        return llm_manager.get_model_for_task("general")
    else:
        return llm_manager.get_model_for_task("general")