"""
src/nodes.py
============
Defines the two LangGraph node functions used in the conversation graph:
  - context_aware_llm : the reasoning node (calls the LLM, decides to use tools or respond)
  - enhanced_tool_node: the execution node (runs whatever tools the LLM requested)

Also contains get_task_optimized_llm(), the keyword-based task classifier that
picks the right model+temperature before each LLM call.

Graph wiring is done in app.py (_build_graph). State schema is in models.py.
"""

from typing import List
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from .models import ConversationState, EnhancedLLM


def create_enhanced_nodes(tools: List, llm_manager: EnhancedLLM):
    """
    Returns (context_aware_llm, enhanced_tool_node) as a tuple of callables.

    Both functions are closures that share the same `tools` list and `llm_manager`
    instance, so they don't need to receive them through graph state on every call.

    Parameters
    ----------
    tools       : list of LangChain tool objects (built by initialize_tools in tools.py)
    llm_manager : EnhancedLLM instance (shared across all graph calls for the session)

    Returns
    -------
    (context_aware_llm, enhanced_tool_node) — pass directly to StateGraph.add_node()
    """

    def context_aware_llm(state: ConversationState) -> ConversationState:
        """
        Core reasoning node. Called by LangGraph on every turn and after each tool run.

        Steps:
          1. Detect task type from the last human message (keyword matching)
          2. Get the right model+temperature for that task
          3. Rebuild the system message with current conversation_summary + last_tool_used
          4. Invoke the LLM with all tools bound
          5. Return the response (may contain tool call requests — LangGraph routes accordingly)

        On failure:
          - Cascades through all 5 models in order, skipping the one that just failed
          - Handles 429 rate limits (daily token exhaustion) by automatically switching models
          - Returns a user-friendly message only when every model is exhausted

        State keys written: messages, error_count, current_model_used
        """
        # Initialize messages before try so it's always accessible in except
        messages = state.get("messages", [])
        try:

            # Scan backwards for the last HumanMessage. Tool messages and AI messages
            # may have been added between the human turn and this node invocation.
            last_human_msg = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_human_msg = msg.content
                    break

            llm = (
                get_task_optimized_llm(llm_manager, last_human_msg)
                if last_human_msg
                else llm_manager.get_llm()
            )

            # Tools must be re-bound every call because the LLM instance is recreated
            # each time (task-specific temperature varies between turns).
            llm_with_tools = llm.bind_tools(tools)

            # System message is rebuilt every turn so conversation_summary and
            # last_tool_used always reflect current state. If we only built it once
            # (on the first turn) those fields would be frozen as empty strings forever.
            # To add new context variables to the prompt, add them to ConversationState
            # (models.py) and reference them here with state.get().
            system_prompt = f"""You are an advanced AI research assistant with access to multiple tools.

Conversation so far: {state.get('conversation_summary', 'New conversation')}
Last tool used: {state.get('last_tool_used', 'None')}

Available tools and when to use them:
- arxiv: Academic papers on physics, math, computer science, engineering
- pub_med: Medical, biomedical, life sciences, and healthcare research
- semantic_scholar_search: Highly-cited papers across all fields with citation counts
- openalex_search: Open access scholarly works across all disciplines
- wikipedia: General knowledge, definitions, background information
- duckduckgo_search (or tavily_search_results_json): Current news, recent events, real-time web info
- calculator: All math — always call this for any numerical computation, never compute in your head
- code_analyzer: Code review and syntax checking
- weather_info: Weather by city name
- file_content_generator: Generate CSV, JSON, Python, or Markdown file content

Rules:
1. Use tools proactively — prefer live tool results over your training knowledge for facts and research
2. For research questions: start with arxiv or pub_med, fall back to semantic_scholar_search
3. For any calculation: always use calculator
4. For current events or news: use duckduckgo_search
5. Cite sources (titles, authors, URLs) when using research tools
6. If one tool fails or returns no results, try an alternative (e.g. arxiv -> semantic_scholar_search)
7. Be thorough but concise — summarize key findings rather than dumping raw tool output"""

            # Replace any existing SystemMessage rather than appending a duplicate.
            # MemorySaver persists all messages across turns, so without this replacement
            # the original system message from turn 1 would remain at position 0 forever
            # with stale (empty) context values.
            non_system = [m for m in messages if not isinstance(m, SystemMessage)]
            messages = [SystemMessage(content=system_prompt)] + non_system

            response = llm_with_tools.invoke(messages)
            current_model = llm_manager.get_current_model_name()
            print(f"Using model: {current_model}")

            return {
                "messages": [response],
                "error_count": 0,             # reset on any successful call
                "current_model_used": current_model,
            }

        except Exception as e:
            print(f"LLM error: {str(e)}")
            error_count = state.get("error_count", 0) + 1
            last_error = str(e)

            # Cascade through every model in the fallback chain. This handles 429
            # rate limits (daily token exhaustion) where the primary is dead for hours
            # — we skip it and try the next model automatically.
            failed_model = llm_manager.get_current_model_name()
            all_configs = (
                [llm_manager.primary_config, llm_manager.secondary_config]
                + llm_manager.fallback_configs
            )

            for config in all_configs:
                if config.name == failed_model:
                    continue
                try:
                    fallback_llm = llm_manager._create_llm_instance(config)
                    if not fallback_llm:
                        continue
                    response = fallback_llm.bind_tools(tools).invoke(messages)
                    llm_manager.current_config = config
                    print(f"Switched to fallback model: {config.name}")
                    return {
                        "messages": [response],
                        "error_count": 0,
                        "current_model_used": config.name,
                    }
                except Exception as fe:
                    last_error = str(fe)
                    print(f"Fallback {config.name} also failed: {last_error}")
                    continue

            # Every model failed — surface a clear message
            is_rate_limit = "429" in last_error or "rate limit" in last_error.lower()
            user_msg = (
                "All models are currently rate-limited. Please wait a few minutes and try again."
                if is_rate_limit
                else f"All models failed. Last error: {last_error}"
            )
            return {
                "messages": [AIMessage(content=user_msg)],
                "error_count": error_count,
            }

    def enhanced_tool_node(state: ConversationState) -> ConversationState:
        """
        Tool execution node. Dispatches the tool calls requested by the LLM
        and returns their results as ToolMessages.

        LangGraph's built-in ToolNode handles the actual dispatch and result formatting.
        This wrapper adds:
          - last_tool_used tracking (written to state, surfaced in next system prompt)
          - error_count increment on failure
          - a user-friendly error message if the tool crashes

        State keys written: messages, last_tool_used, error_count

        To add a new tool: define it in tools.py, add it to initialize_tools(), and
        add its name + description to the system prompt in context_aware_llm above.
        No changes needed here.
        """
        try:
            result = ToolNode(tools).invoke(state)

            messages = result.get("messages", [])
            if messages:
                # Track last tool for the system prompt context on the next turn
                tool_name = getattr(messages[-1], "name", "unknown_tool")
                result.update({
                    "messages": messages,
                    "last_tool_used": tool_name,
                    "error_count": 0,
                })

            return result

        except Exception as e:
            print(f"Tool execution error: {str(e)}")
            return {
                "messages": [AIMessage(
                    content=(
                        f"I encountered an error while using the tools: {str(e)}. "
                        "Please try rephrasing your question."
                    )
                )],
                "error_count": state.get("error_count", 0) + 1,
            }

    return context_aware_llm, enhanced_tool_node


def get_task_optimized_llm(llm_manager: EnhancedLLM, user_message: str):
    """
    Keyword-based task classifier. Maps the user's message to one of the task
    types defined in TASK_TEMPERATURES (models.py) so the right model and
    temperature are applied.

    To add a new task type:
      1. Add the type + temperature to TASK_TEMPERATURES in models.py
      2. Add it to task_model_mapping in EnhancedLLM.get_model_for_task() in models.py
      3. Add keyword detection here

    Simple keyword matching is intentional — an LLM-based classifier would add a
    full inference round-trip just to route the request, which is not worth it.

    Parameters
    ----------
    llm_manager  : EnhancedLLM instance
    user_message : the raw user input string

    Returns
    -------
    ChatGroq instance configured for the detected task type
    """
    msg = user_message.lower()

    if any(w in msg for w in ["calculate", "math", "equation", "solve", "compute", "%", "percent"]):
        return llm_manager.get_model_for_task("math")
    elif any(w in msg for w in ["research", "paper", "study", "academic", "arxiv", "latest research"]):
        return llm_manager.get_model_for_task("reasoning")
    elif any(w in msg for w in ["analyze", "analysis", "compare", "evaluate"]):
        return llm_manager.get_model_for_task("analysis")
    elif any(w in msg for w in ["code", "programming", "python", "javascript", "debug"]):
        return llm_manager.get_model_for_task("coding")
    elif any(w in msg for w in ["write", "story", "poem", "creative", "imagine"]):
        return llm_manager.get_model_for_task("creative")
    else:
        return llm_manager.get_model_for_task("general")
