"""
src/nodes.py
============
Multi-agent LangGraph nodes.

Agents:
  supervisor_node    - keyword-based router (no LLM call, fast)
  research_agent     - handles academic search, web, code, math queries
  pdf_agent          - handles uploaded PDF analysis + literature connection

Tool nodes (each agent has its own isolated ToolNode):
  research_tool_node - all tools except pdf_search
  pdf_tool_node      - pdf_search + arxiv + semantic_scholar + find_related_papers

Public factory:
  create_agent_nodes(tools, llm_manager)
    -> supervisor_node, route_after_supervisor,
       research_agent, pdf_agent,
       research_tool_node, pdf_tool_node

Helper:
  get_task_optimized_llm(llm_manager, user_message)
    -> keyword-based temperature + model routing
"""

from typing import List, Callable
from langchain_core.messages import (
    AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage,
)
from langgraph.prebuilt import ToolNode, tools_condition
from .models import ConversationState, EnhancedLLM
from .tools import _active_pdf_names


# ---------------------------------------------------------------------------
# Supervisor — keyword routing, no LLM call
# ---------------------------------------------------------------------------

def supervisor_node(state: ConversationState) -> dict:
    """
    Routes to research_agent or pdf_agent without calling an LLM.

    Priority:
      1. PDF keywords present + papers loaded  -> pdf_agent
      2. Academic search keywords present      -> research_agent
      3. Papers loaded, ambiguous query        -> pdf_agent  (default with papers)
      4. No papers, no clear signal            -> research_agent
    """
    messages = state.get("messages", [])
    msg = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            msg = m.content.lower()
            break

    pdf_names = _active_pdf_names.get()
    has_pdfs  = bool(pdf_names)

    pdf_keywords = [
        "pdf", "paper", "document", "uploaded", "my paper", "this paper",
        "the paper", "what does it say", "summarize this", "in the paper",
        "according to", "the study", "this study", "what i uploaded",
        "the file", "the document", "this document",
    ]
    research_keywords = [
        "find papers", "latest research", "search for", "arxiv", "pubmed",
        "recent studies", "academic", "semantic scholar", "find related",
        "search papers", "literature review",
    ]

    if has_pdfs and any(kw in msg for kw in pdf_keywords):
        agent = "pdf"
    elif any(kw in msg for kw in research_keywords):
        agent = "research"
    elif has_pdfs:
        # Papers loaded but query is ambiguous — pdf_agent by default
        agent = "pdf"
    else:
        agent = "research"

    print(f"[supervisor] -> {agent}_agent (has_pdfs={has_pdfs})")
    return {"current_agent": agent, "active_pdfs": list(pdf_names)}


def route_after_supervisor(state: ConversationState) -> str:
    """LangGraph conditional edge: maps current_agent to node name."""
    return state.get("current_agent", "research") + "_agent"


# ---------------------------------------------------------------------------
# System prompts — one per agent
# ---------------------------------------------------------------------------

def _research_prompt(state: ConversationState, pdf_ctx: str) -> str:
    return f"""You are an elite AI research intelligence assistant — expert academic researcher, \
data scientist, and knowledge synthesizer.

ACTIVE CONTEXT:
- Conversation: {state.get('conversation_summary', 'New conversation')}
- Last tool used: {state.get('last_tool_used', 'None')}
- {pdf_ctx}

TOOL SELECTION (follow this priority exactly):
CRITICAL RULE: Call each tool AT MOST ONCE per response. Never repeat a tool call.

1. ACADEMIC RESEARCH — pick the most relevant 1-2 sources:
   - arxiv                    -> CS, math, physics, engineering preprints
   - pub_med                  -> medical, biomedical, clinical trials
   - semantic_scholar_search  -> citation counts, cross-field impact
   - openalex_search          -> open-access works across all disciplines
   - find_related_papers      -> arXiv + Semantic Scholar in one call
     (use this INSTEAD of calling arxiv AND semantic_scholar_search separately)

2. CALCULATIONS -> ALWAYS call calculator. Show the formula first.

3. CODE QUESTIONS -> use code_analyzer. Include Python code blocks.

4. CURRENT EVENTS -> duckduckgo_search or tavily_search_results_json.

5. GENERAL KNOWLEDGE -> wikipedia for background.

RESPONSE QUALITY:
- Synthesize findings — do not just summarize tool output
- Explain WHY researchers chose specific approaches (methodology rationale)
- Connect findings to the research landscape: what came before, what this builds on
- Identify consensus vs. active debates in the field
- Suggest future research directions and open problems
- Use Mermaid diagrams (```mermaid) for system architecture
- Use LaTeX ($formula$) for math notation

CITATIONS:
- External papers: (Author et al., Year)
- Web sources: [Source: domain]
When you used a tool OR made factual/research claims, end with:
## References
1. Author(s). "Title." Venue, Year.

For greetings or simple questions: do NOT add a References section."""


def _pdf_prompt(state: ConversationState, pdf_ctx: str) -> str:
    return f"""You are a research paper analysis specialist with deep expertise in academic literature.

ACTIVE CONTEXT:
- Conversation: {state.get('conversation_summary', 'New conversation')}
- Last tool used: {state.get('last_tool_used', 'None')}
- {pdf_ctx}

YOUR MISSION: Help the user deeply understand their uploaded research paper and
connect it to the broader academic landscape.

TOOL SELECTION:
CRITICAL RULE: Call each tool AT MOST ONCE per response. Never repeat a tool call.

1. ALWAYS call pdf_search first to ground the answer in the actual paper.
   Cite every claim as [Paper: title, Page: N].

2. After answering from the paper, call find_related_papers OR arxiv ONCE to
   connect it to external literature. Do not call both.

3. Call semantic_scholar_search only if you specifically need citation counts.

ANALYSIS DEPTH — address all of these in every answer:
- What does the paper say? (direct citations with page numbers)
- What prior work does this build on? (connect to external literature)
- Why did the authors choose this approach over alternatives?
- What are the key findings, limitations, and future directions?

RESPONSE STRUCTURE (use these sections):
- **From the Paper** — direct quotes/paraphrases with [Paper: title, Page: N]
- **Research Context** — how this relates to prior or concurrent work
- **Methodology Rationale** — why this approach, what alternatives exist
- **Limitations and Future Work** — what the authors acknowledge as gaps

CITATIONS:
- Uploaded PDFs: [Paper: title, Page: N]
- External papers: (Author et al., Year)
Always end with a ## References section when tools were used."""


# ---------------------------------------------------------------------------
# Generic LLM node factory (shared by both agents)
# ---------------------------------------------------------------------------

def _make_llm_node(
    tools: List,
    llm_manager: EnhancedLLM,
    get_prompt: Callable[[ConversationState, str], str],
    agent_label: str,
) -> Callable:
    """
    Builds an LLM reasoning node for one agent.

    Parameters
    ----------
    tools       : tools this agent can call (already filtered to agent's scope)
    llm_manager : shared LLM manager with fallback chain
    get_prompt  : function(state, pdf_ctx) -> system prompt string
    agent_label : "research" or "pdf" — for logging only
    """

    def node(state: ConversationState) -> dict:
        messages = state.get("messages", [])

        try:
            last_human = next(
                (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
                None,
            )
            llm = (
                get_task_optimized_llm(llm_manager, last_human)
                if last_human
                else llm_manager.get_llm()
            )
            llm_with_tools = llm.bind_tools(tools)

            # Build pdf context string injected into the system prompt every turn
            pdf_names = _active_pdf_names.get()
            if pdf_names:
                pdf_ctx = (
                    "UPLOADED PAPERS (use pdf_search to search these):\n"
                    + "\n".join(f"  - {n}" for n in pdf_names)
                )
            else:
                pdf_ctx = "No papers currently uploaded."

            system_prompt = get_prompt(state, pdf_ctx)

            # Replace existing SystemMessage each turn so the prompt is always fresh
            non_system = [m for m in messages if not isinstance(m, SystemMessage)]
            full_messages = [SystemMessage(content=system_prompt)] + non_system

            response      = llm_with_tools.invoke(full_messages)
            current_model = llm_manager.get_current_model_name()
            print(f"[{agent_label}_agent] {llm_manager.get_provider()}/{current_model}")

            return {
                "messages":           [response],
                "error_count":        0,
                "current_model_used": current_model,
            }

        except Exception as e:
            print(f"[{agent_label}_agent] LLM error: {e}")
            error_count = state.get("error_count", 0) + 1
            last_error  = str(e)

            # Walk fallback chain — skip the model that just failed
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
                    # Groq 8b has a 6000 TPM cap — trim history to avoid 413
                    msgs_to_send = messages
                    if config.provider == "groq" and config.max_tokens <= 2048:
                        msgs_to_send = messages[:1] + messages[-3:]
                    response = fallback_llm.bind_tools(tools).invoke(msgs_to_send)
                    llm_manager.current_config = config
                    print(f"[{agent_label}_agent] Switched to fallback: {config.name}")
                    return {
                        "messages":           [response],
                        "error_count":        0,
                        "current_model_used": config.name,
                    }
                except Exception as fe:
                    last_error = str(fe)
                    print(f"[{agent_label}_agent] Fallback {config.name} failed: {fe}")
                    continue

            is_rate_limit = "429" in last_error or "rate limit" in last_error.lower()
            user_msg = (
                "All models are currently rate-limited. Please wait a few minutes."
                if is_rate_limit
                else f"All models failed. Last error: {last_error}"
            )
            return {
                "messages":    [AIMessage(content=user_msg)],
                "error_count": error_count,
            }

    return node


# ---------------------------------------------------------------------------
# Tool node factory — wraps ToolNode with graceful error handling
# ---------------------------------------------------------------------------

def _make_tool_node(tools: List) -> Callable:
    """Wraps LangGraph's ToolNode; converts crashes into recovery ToolMessages."""

    def node(state: ConversationState) -> dict:
        try:
            result = ToolNode(tools).invoke(state)
            messages = result.get("messages", [])
            if messages:
                result.update({
                    "messages":       messages,
                    "last_tool_used": getattr(messages[-1], "name", "unknown_tool"),
                    "error_count":    0,
                })
            return result

        except Exception as e:
            print(f"[tool_node] Error: {e}")
            messages    = state.get("messages", [])
            last_ai_msg = next(
                (m for m in reversed(messages)
                 if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)),
                None,
            )
            if last_ai_msg:
                return {
                    "messages": [
                        ToolMessage(
                            content=f"Tool error: {e}. Try a different approach.",
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                        for tc in last_ai_msg.tool_calls
                    ],
                    "error_count": state.get("error_count", 0) + 1,
                }
            return {
                "messages":    [AIMessage(content=f"Error using tools: {e}")],
                "error_count": state.get("error_count", 0) + 1,
            }

    return node


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_agent_nodes(tools: List, llm_manager: EnhancedLLM):
    """
    Build all nodes for the multi-agent graph.

    Tool split:
      research_tools: everything except pdf_search
        -> handles web, academic APIs, calculator, code, weather, file gen
      pdf_tools: pdf_search + arxiv + semantic_scholar + find_related_papers
        -> handles uploaded PDF queries + external literature connection

    Returns
    -------
    (supervisor_node, route_after_supervisor,
     research_agent, pdf_agent,
     research_tool_node, pdf_tool_node)
    """
    PDF_TOOL_NAMES = {"pdf_search", "arxiv", "semantic_scholar_search", "find_related_papers"}

    research_tools = [t for t in tools if getattr(t, "name", "") not in {"pdf_search"}]
    pdf_tools      = [t for t in tools if getattr(t, "name", "") in PDF_TOOL_NAMES]

    research_agent     = _make_llm_node(research_tools, llm_manager, _research_prompt, "research")
    pdf_agent          = _make_llm_node(pdf_tools,      llm_manager, _pdf_prompt,      "pdf")
    research_tool_node = _make_tool_node(research_tools)
    pdf_tool_node      = _make_tool_node(pdf_tools)

    return (
        supervisor_node,
        route_after_supervisor,
        research_agent,
        pdf_agent,
        research_tool_node,
        pdf_tool_node,
    )


# ---------------------------------------------------------------------------
# Task-based LLM routing (shared by both agents)
# ---------------------------------------------------------------------------

def get_task_optimized_llm(llm_manager: EnhancedLLM, user_message: str):
    """
    Keyword-based task classifier. Maps the user message to a task type and
    returns an LLM with the appropriate temperature.

    Simple keyword matching is intentional — an LLM-based classifier would add
    a full inference round-trip just to route the request.
    """
    msg = user_message.lower()

    if any(w in msg for w in ["calculate", "math", "equation", "solve", "compute",
                               "integral", "derivative", "percent", "%"]):
        return llm_manager.get_model_for_task("math")
    elif any(w in msg for w in ["research", "paper", "study", "academic", "arxiv",
                                 "pubmed", "literature", "latest research", "find papers"]):
        return llm_manager.get_model_for_task("reasoning")
    elif any(w in msg for w in ["analyze", "analysis", "compare", "evaluate",
                                 "review", "assess", "critique"]):
        return llm_manager.get_model_for_task("analysis")
    elif any(w in msg for w in ["code", "programming", "python", "javascript",
                                 "function", "debug", "implement", "algorithm"]):
        return llm_manager.get_model_for_task("coding")
    elif any(w in msg for w in ["write", "story", "poem", "creative",
                                 "imagine", "generate text", "essay"]):
        return llm_manager.get_model_for_task("creative")
    else:
        return llm_manager.get_model_for_task("general")
