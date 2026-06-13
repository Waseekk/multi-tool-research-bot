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
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from .models import ConversationState, EnhancedLLM
from .tools import _active_pdf_names


def create_enhanced_nodes(tools: List, llm_manager: EnhancedLLM):
    """
    Returns (context_aware_llm, enhanced_tool_node) as a tuple of callables.

    Both functions are closures that share the same `tools` list and `llm_manager`
    instance so they don't need to receive them through graph state on every call.
    """

    def context_aware_llm(state: ConversationState) -> ConversationState:
        """
        Core reasoning node. Called by LangGraph on every turn and after each tool run.

        Steps:
          1. Detect task type from the last human message (keyword matching)
          2. Get the right model+temperature for that task
          3. Rebuild the system message with current pdf context + conversation summary
          4. Invoke the LLM with all tools bound
          5. Return the response (may contain tool call requests — LangGraph routes accordingly)

        On failure: cascades through the full model fallback chain, then surfaces a
        user-friendly message only when every model is exhausted.
        """
        messages = state.get("messages", [])
        try:
            # Scan backwards for the last HumanMessage to detect task type
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

            llm_with_tools = llm.bind_tools(tools)

            # Build uploaded-papers context — injected into system prompt each turn
            # so the agent always knows what PDF files are available for pdf_search
            pdf_names = _active_pdf_names.get()
            if pdf_names:
                pdf_ctx = (
                    "UPLOADED PAPERS (call pdf_search to search these — do not skip this step):\n"
                    + "\n".join(f"  • {n}" for n in pdf_names)
                )
            else:
                pdf_ctx = "No papers currently uploaded."

            # The system prompt is the primary lever for response quality.
            # It is rebuilt every turn so pdf_ctx / conversation_summary always reflect
            # the current state rather than the values from the very first turn.
            system_prompt = f"""You are an elite AI research intelligence assistant — an expert academic researcher, \
data scientist, and knowledge synthesizer.

ACTIVE CONTEXT:
• Conversation history: {state.get('conversation_summary', 'New conversation')}
• Last tool used: {state.get('last_tool_used', 'None')}
• {pdf_ctx}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL SELECTION — follow this priority order exactly
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PDF QUERIES — user mentions "my paper", "the document", "uploaded", "this study", \
or papers are listed above → call pdf_search FIRST, then find_related_papers to connect \
it to the broader field. Cite as [Paper: title, Page: N].

2. ACADEMIC RESEARCH — use multiple sources for comprehensive coverage:
   • arxiv → CS, math, physics, engineering (preprints + published)
   • pub_med → medical, biomedical, neuroscience, clinical trials
   • semantic_scholar_search → citation counts, highly-cited works across all fields
   • find_related_papers → simultaneous arXiv + Semantic Scholar search in one call
   Use at least 2 sources for any research topic; cross-reference findings.

3. CALCULATIONS → ALWAYS call calculator. Never compute in your head. Show the formula first.

4. CODE QUESTIONS → use code_analyzer for debugging/review. Include Python code blocks in answers.

5. CURRENT EVENTS / NEWS → duckduckgo_search or tavily_search_results_json.

6. GENERAL KNOWLEDGE → wikipedia for background, then web search for recent developments.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE QUALITY STANDARDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEPTH OF ANALYSIS:
• Synthesize findings — do not just summarize tool output
• Explain WHY researchers chose specific approaches (methodology rationale)
• Connect findings to the research landscape: what came before, what this builds on
• Identify consensus vs. active debates in the field
• Point out methodological strengths and limitations
• Suggest future research directions and open problems

STRUCTURE — use these sections where relevant:
• **Key Findings** — the core answer, synthesized clearly
• **Methodology & Rationale** — how the research was done and why this approach
• **Research Context** — prior work this builds on; historical evolution
• **Debate & Open Questions** — what is still contested or unknown
• **Future Directions** — emerging areas and next research steps

CODE & DIAGRAMS — include whenever explaining technical concepts:
• Algorithms: always provide a Python code example
• System architecture: use Mermaid flowcharts inside ```mermaid blocks
• Data flows: ASCII diagrams are acceptable when Mermaid is not natural
• Mathematical notation: use LaTeX-style inline ($formula$) or block ($$formula$$)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATIONS & REFERENCES (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every factual claim must be cited in-text:
• Uploaded PDFs → [Paper: filename, Page: N]
• External papers → (Author et al., Year)
• Web sources → [Source: title or domain]

EVERY response that makes factual claims or uses any tool MUST end with a References section:

## References
1. Author(s). "Title." Venue, Year. DOI or URL if available.
2. ...

If the answer is from general knowledge with no tools used:
## References
*Based on general knowledge. For verified details, consider checking primary sources.*"""

            # Replace any existing SystemMessage rather than appending a duplicate.
            # MemorySaver persists all messages across turns, so without this the
            # original system message from turn 1 stays frozen with stale context values.
            non_system = [m for m in messages if not isinstance(m, SystemMessage)]
            messages   = [SystemMessage(content=system_prompt)] + non_system

            response      = llm_with_tools.invoke(messages)
            current_model = llm_manager.get_current_model_name()
            print(f"[nodes] Response from {llm_manager.get_provider()}/{current_model}")

            return {
                "messages":           [response],
                "error_count":        0,
                "current_model_used": current_model,
            }

        except Exception as e:
            print(f"[nodes] LLM error: {e}")
            error_count = state.get("error_count", 0) + 1
            last_error  = str(e)

            # Cascade through every fallback model. Handles 429 rate limits (daily
            # token exhaustion) where the primary is dead for hours — we skip it and
            # try the next model automatically.
            failed_model = llm_manager.get_current_model_name()
            all_configs  = (
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
                    print(f"[nodes] Switched to fallback: {config.provider}/{config.name}")
                    return {
                        "messages":           [response],
                        "error_count":        0,
                        "current_model_used": config.name,
                    }
                except Exception as fe:
                    last_error = str(fe)
                    print(f"[nodes] Fallback {config.name} also failed: {fe}")
                    continue

            is_rate_limit = "429" in last_error or "rate limit" in last_error.lower()
            user_msg = (
                "All models are currently rate-limited. Please wait a few minutes and try again."
                if is_rate_limit
                else f"All models failed. Last error: {last_error}"
            )
            return {
                "messages":   [AIMessage(content=user_msg)],
                "error_count": error_count,
            }

    def enhanced_tool_node(state: ConversationState) -> ConversationState:
        """
        Tool execution node. Dispatches the tool calls requested by the LLM
        and returns their results as ToolMessages.

        Adds last_tool_used tracking (surfaced in next system prompt) and wraps
        crashes in a user-friendly ToolMessage so the LLM can recover gracefully.
        """
        try:
            result = ToolNode(tools).invoke(state)

            messages = result.get("messages", [])
            if messages:
                tool_name = getattr(messages[-1], "name", "unknown_tool")
                result.update({
                    "messages":      messages,
                    "last_tool_used": tool_name,
                    "error_count":   0,
                })

            return result

        except Exception as e:
            print(f"[nodes] Tool execution error: {e}")
            messages     = state.get("messages", [])
            last_ai_msg  = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    last_ai_msg = msg
                    break

            if last_ai_msg and last_ai_msg.tool_calls:
                return {
                    "messages": [
                        ToolMessage(
                            content=f"Tool error: {e}. Try a different approach or tool.",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                        for tc in last_ai_msg.tool_calls
                    ],
                    "error_count": state.get("error_count", 0) + 1,
                }
            return {
                "messages":   [AIMessage(content=f"Error using tools: {e}")],
                "error_count": state.get("error_count", 0) + 1,
            }

    return context_aware_llm, enhanced_tool_node


def get_task_optimized_llm(llm_manager: EnhancedLLM, user_message: str):
    """
    Keyword-based task classifier. Maps the user's message to one of the task
    types in TASK_TEMPERATURES (models.py) and returns the appropriately configured LLM.

    Simple keyword matching is intentional — an LLM-based classifier would add a
    full inference round-trip just to route the request, which is not worth the cost.
    """
    msg = user_message.lower()

    if any(w in msg for w in ["calculate", "math", "equation", "solve", "compute", "integral", "derivative", "%", "percent"]):
        return llm_manager.get_model_for_task("math")
    elif any(w in msg for w in ["research", "paper", "study", "academic", "arxiv", "pubmed", "literature", "latest research", "find papers"]):
        return llm_manager.get_model_for_task("reasoning")
    elif any(w in msg for w in ["analyze", "analysis", "compare", "evaluate", "review", "assess", "critique"]):
        return llm_manager.get_model_for_task("analysis")
    elif any(w in msg for w in ["code", "programming", "python", "javascript", "function", "debug", "implement", "algorithm"]):
        return llm_manager.get_model_for_task("coding")
    elif any(w in msg for w in ["write", "story", "poem", "creative", "imagine", "generate text", "essay"]):
        return llm_manager.get_model_for_task("creative")
    else:
        return llm_manager.get_model_for_task("general")
