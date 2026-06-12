"""
src/models.py
=============
Defines the shared state schema (ConversationState) and the LLM management layer
(EnhancedLLM). Everything that touches Groq model selection or fallback lives here.

Graph nodes (nodes.py) import EnhancedLLM and ConversationState.
app.py imports both to build the LangGraph graph.
"""

from typing import List, Dict, Any, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ConversationState(TypedDict):
    """
    The single state object passed between every LangGraph node.

    `messages` uses the add_messages reducer so new messages are appended rather
    than replacing the list — this is what accumulates multi-turn history in
    MemorySaver. All other fields use simple last-write-wins assignment.

    Fields
    ------
    messages            : full message history (HumanMessage, AIMessage, ToolMessage, SystemMessage)
    user_context        : reserved for future per-user preferences / metadata
    tool_results_cache  : reserved; currently unused — placeholder for caching repeated queries
    conversation_summary: last 3 user questions joined by " | "; injected into system prompt each turn
                          (built by ConversationManager.summarize_conversation in conversation.py)
    error_count         : increments on LLM/tool failure; resets to 0 on success; triggers fallback at 1
    last_tool_used      : name of the most recently executed tool; surfaced in the system prompt
    current_model_used  : name of the Groq model that produced the last AI response
    model_switch_count  : how many times the fallback chain was triggered in this session
    """
    messages: Annotated[List[AnyMessage], add_messages]
    user_context: Dict[str, Any]
    tool_results_cache: Dict[str, Any]
    conversation_summary: str
    error_count: int
    last_tool_used: str
    current_model_used: str
    model_switch_count: int


# ---------------------------------------------------------------------------
# Task temperatures
# ---------------------------------------------------------------------------

# Controls how deterministic vs. creative each task type is.
# To add a new task type: add a key here and a matching entry in the
# task_model_mapping dict inside EnhancedLLM.get_model_for_task().
TASK_TEMPERATURES = {
    "math":      0.0,   # must be deterministic — any randomness risks wrong answers
    "coding":    0.0,   # same reason: code correctness is not probabilistic
    "analysis":  0.05,  # near-deterministic but slight flexibility for framing
    "reasoning": 0.1,   # low temperature; we want coherent chains of thought
    "general":   0.1,
    "creative":  0.7,   # higher entropy produces more varied, interesting text
}


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

class ModelConfig:
    """
    Configuration + runtime health stats for one Groq model.

    failure_count and last_failure_time are written by _create_llm_instance when
    an instantiation fails, and read by _is_model_available to enforce the cooldown.
    success_count is informational only (not used in routing logic).
    """

    def __init__(self, name: str, temperature: float = 0.1, max_tokens: int = 4096, max_retries: int = 2):
        self.name = name
        self.temperature = temperature   # default; overridden per-task by get_model_for_task()
        self.max_tokens = max_tokens     # output token cap — 4096 fits long research summaries
        self.max_retries = max_retries
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0


# ---------------------------------------------------------------------------
# LLM manager
# ---------------------------------------------------------------------------

class EnhancedLLM:
    """
    Manages a cascade of Groq models with automatic failover and per-task routing.

    Fallback chain (tried in order when the preferred model is unavailable):
        1. llama-3.3-70b-versatile             — primary; best overall quality
        2. llama-3.1-8b-instant                — secondary; fast, confirmed tool-calling support
        3. llama-3.1-8b-instant                — fallback 1 (same model, different instance)
        4. llama3-groq-8b-8192-tool-use-preview — last resort; fine-tuned specifically for tool use

    NOTE: llama-3.1-70b-versatile was decommissioned by Groq on ~2026-06-11 and removed.
    It was previously the secondary and caused every research/analysis query to immediately
    fail with HTTP 400, since task_model_mapping routes those tasks to secondary_config.
    Only models with confirmed Groq tool-calling support are included.

    A failed model enters a 60-second cooldown before being retried, preventing
    repeated hammering of a rate-limited endpoint while still recovering automatically.

    Usage (from nodes.py):
        llm_manager = EnhancedLLM()
        llm = llm_manager.get_model_for_task("math")   # task-specific routing
        llm = llm_manager.get_llm()                     # plain fallback chain
    """

    def __init__(self):
        self.primary_config   = ModelConfig("llama-3.3-70b-versatile", temperature=0.1, max_tokens=4096)
        # llama-3.1-70b-versatile was decommissioned by Groq (~2026-06-11).
        # Using llama-3.1-8b-instant instead — smaller but confirmed working and supports tool calling.
        self.secondary_config = ModelConfig("llama-3.1-8b-instant",    temperature=0.1, max_tokens=2048)

        # Only models with confirmed Groq tool-calling support are included here.
        # gemma2-9b-it and llama3-70b-8192 were removed: they ignore bind_tools() on Groq,
        # causing the agent to stall silently when rate limits push the app to those fallbacks.
        self.fallback_configs = [
            ModelConfig("llama-3.1-8b-instant",                temperature=0.1, max_tokens=2048),
            ModelConfig("llama3-groq-8b-8192-tool-use-preview", temperature=0.1, max_tokens=2048),
        ]

        self.current_config = self.primary_config
        self.use_secondary = False   # flips to True after primary fails; resets on manual call
        self.total_requests = 0
        self.cooldown_period = 60    # seconds before a failed model is retried

    def _is_model_available(self, config: ModelConfig) -> bool:
        """Returns False if the model is within the post-failure cooldown window."""
        if config.last_failure_time is None:
            return True
        return (time.time() - config.last_failure_time) > self.cooldown_period

    def _create_llm_instance(self, config: ModelConfig, temperature: Optional[float] = None) -> Optional[ChatGroq]:
        """
        Creates a ChatGroq client. Returns None on failure and records the failure
        time so the cooldown system can skip this model on subsequent calls.

        No test/probe call is made here. Previously a "Say OK" test was sent on every
        instantiation, which doubled API usage and added ~1-2s latency per turn.
        Real failures surface at invoke() time and are caught by context_aware_llm.

        Parameters
        ----------
        config      : which model and its default settings
        temperature : overrides config.temperature when set (used by get_model_for_task)
        """
        try:
            llm = ChatGroq(
                model=config.name,
                temperature=temperature if temperature is not None else config.temperature,
                max_tokens=config.max_tokens,
            )
            config.success_count += 1
            logger.info(f"Initialized model: {config.name}")
            return llm
        except Exception as e:
            config.failure_count += 1
            config.last_failure_time = time.time()
            logger.warning(f"Failed to initialize model {config.name}: {str(e)}")
        return None

    def get_llm(self, prefer_secondary: bool = False, force_model: Optional[str] = None) -> ChatGroq:
        """
        Returns the best available LLM, walking the fallback chain if needed.

        Call flow:
          1. If force_model is set, jump directly to that model.
          2. Try primary + secondary in preference order (skipping models in cooldown).
          3. Try fallback_configs in order.
          4. If everything is in cooldown, retry without cooldown restrictions.
          5. Raise if nothing works.

        Raises
        ------
        Exception  if every model fails even without cooldown restrictions
        """
        self.total_requests += 1

        if force_model:
            for config in [self.primary_config, self.secondary_config] + self.fallback_configs:
                if config.name == force_model:
                    llm = self._create_llm_instance(config)
                    if llm:
                        self.current_config = config
                        return llm
                    break

        primary_order = (
            [self.secondary_config, self.primary_config]
            if prefer_secondary or self.use_secondary
            else [self.primary_config, self.secondary_config]
        )

        for config in primary_order:
            if self._is_model_available(config):
                llm = self._create_llm_instance(config)
                if llm:
                    self.current_config = config
                    self.use_secondary = (config == self.secondary_config)
                    return llm

        logger.warning("Primary and secondary in cooldown, trying fallback models...")
        for config in self.fallback_configs:
            if self._is_model_available(config):
                llm = self._create_llm_instance(config)
                if llm:
                    self.current_config = config
                    logger.info(f"Using fallback model: {config.name}")
                    return llm

        # Last resort: ignore cooldown and try everything once more
        logger.error("All models in cooldown — retrying without restriction...")
        for config in [self.primary_config, self.secondary_config] + self.fallback_configs:
            llm = self._create_llm_instance(config)
            if llm:
                self.current_config = config
                logger.info(f"Emergency fallback to: {config.name}")
                return llm

        raise Exception("All models failed. Check GROQ_API_KEY and internet connection.")

    def get_primary_llm(self, **kwargs) -> ChatGroq:
        """Explicitly request the primary model (llama-3.3-70b-versatile)."""
        return self.get_llm(prefer_secondary=False, **kwargs)

    def get_secondary_llm(self, **kwargs) -> ChatGroq:
        """Explicitly request the secondary model (llama-3.1-70b-versatile).
        Called by the error handler in context_aware_llm (nodes.py) on first failure."""
        return self.get_llm(prefer_secondary=True, **kwargs)

    def switch_to_secondary(self) -> None:
        """Flip default preference to secondary for the rest of the session."""
        self.use_secondary = True
        logger.info("Switched preference to secondary model")

    def switch_to_primary(self) -> None:
        self.use_secondary = False
        logger.info("Switched preference back to primary model")

    def get_model_for_task(self, task_type: str) -> ChatGroq:
        """
        Selects the model and temperature best suited to task_type.

        Task routing:
          math / coding   -> primary (llama-3.3-70b), temp=0.0  — deterministic output needed
          reasoning       -> secondary (llama-3.1-70b), temp=0.1 — structured open-ended responses
          analysis        -> secondary, temp=0.05
          creative        -> primary, temp=0.7
          general         -> primary, temp=0.1

        To add a new task type: add to TASK_TEMPERATURES and to task_model_mapping below,
        then add the keyword detection in get_task_optimized_llm (nodes.py).

        Falls back to get_llm() if the preferred model is in cooldown.
        """
        task_model_mapping = {
            "reasoning": self.secondary_config,
            "math":      self.primary_config,
            "analysis":  self.secondary_config,
            "coding":    self.primary_config,
            "creative":  self.primary_config,
            "general":   self.primary_config,
        }

        temperature = TASK_TEMPERATURES.get(task_type.lower(), 0.1)
        preferred_config = task_model_mapping.get(task_type.lower(), self.primary_config)

        if self._is_model_available(preferred_config):
            llm = self._create_llm_instance(preferred_config, temperature=temperature)
            if llm:
                self.current_config = preferred_config
                logger.info(f"Using {preferred_config.name} for {task_type} (temp={temperature})")
                return llm

        logger.info(f"Preferred model for '{task_type}' unavailable, falling back")
        return self.get_llm()

    def get_current_model_name(self) -> str:
        """Returns the name of the last model successfully instantiated."""
        return self.current_config.name

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Returns a dict of runtime stats for all models. Useful for debugging
        which models are in cooldown or how often fallbacks are triggered.

        Return shape:
        {
          "total_requests": int,
          "current_model": str,
          "using_secondary_preference": bool,
          "models": {
            "<model_name>": {
              "success_count": int,
              "failure_count": int,
              "last_failure": float | None,   # Unix timestamp
              "available": bool
            }, ...
          }
        }
        """
        all_configs = [self.primary_config, self.secondary_config] + self.fallback_configs
        return {
            "total_requests": self.total_requests,
            "current_model": self.current_config.name,
            "using_secondary_preference": self.use_secondary,
            "models": {
                c.name: {
                    "success_count": c.success_count,
                    "failure_count": c.failure_count,
                    "last_failure": c.last_failure_time,
                    "available": self._is_model_available(c),
                }
                for c in all_configs
            },
        }

    def reset_model_stats(self) -> None:
        """Clears all failure counts and cooldown timers. Useful for testing."""
        for config in [self.primary_config, self.secondary_config] + self.fallback_configs:
            config.success_count = 0
            config.failure_count = 0
            config.last_failure_time = None
        self.total_requests = 0
        logger.info("Model statistics reset")
