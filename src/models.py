"""
src/models.py
=============
Defines the shared state schema (ConversationState) and the LLM management layer
(EnhancedLLM). Supports Anthropic (Claude), OpenAI (GPT-4o), and Groq (Llama) providers.

Provider chain is auto-detected from environment variables (priority order):
  - ANTHROPIC_API_KEY set → Claude Opus 4.8 (primary), Claude Sonnet 4.6 (secondary)
  - OPENAI_API_KEY set    → GPT-4o (primary), GPT-4o-mini (secondary)
  - Only GROQ_API_KEY     → llama-3.3-70b (primary), llama-4-scout (secondary)

Groq models are always added as emergency fallbacks when GROQ_API_KEY is also set.

Graph nodes (nodes.py) import EnhancedLLM and ConversationState.
app.py imports both to build the LangGraph graph.
"""

import os
import time
import logging
from typing import List, Dict, Any, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages

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

TASK_TEMPERATURES = {
    "math":      0.0,   # deterministic — any randomness risks wrong answers
    "coding":    0.0,   # same: code correctness is not probabilistic
    "analysis":  0.05,  # near-deterministic but slight flexibility for framing
    "reasoning": 0.1,   # low temperature for coherent chains of thought
    "general":   0.1,
    "creative":  0.7,   # higher entropy for more varied, interesting text
}


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

class ModelConfig:
    """
    Configuration + runtime health stats for one model.

    provider field selects which SDK to use: "openai" or "groq".
    failure_count / last_failure_time enforce the 60-second post-failure cooldown.
    """

    def __init__(
        self,
        name: str,
        provider: str = "groq",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_retries: int = 2,
    ):
        self.name = name
        self.provider = provider          # "openai" or "groq"
        self.temperature = temperature    # default; overridden per-task by get_model_for_task()
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0


# ---------------------------------------------------------------------------
# LLM manager
# ---------------------------------------------------------------------------

class EnhancedLLM:
    """
    Multi-provider LLM manager with automatic failover and per-task routing.

    Provider chain (auto-built from env vars on __init__, priority order):

    If ANTHROPIC_API_KEY is set:
        1. claude-opus-4-8     (anthropic, primary)   — most capable, best research reasoning
        2. claude-sonnet-4-6   (anthropic, secondary)  — fast, excellent quality
        3. llama-3.3-70b       (groq fallback)         — only if GROQ_API_KEY also set

    If only OPENAI_API_KEY is set:
        1. gpt-4o              (openai, primary)
        2. gpt-4o-mini         (openai, secondary)
        3. llama-3.3-70b       (groq fallback)         — only if GROQ_API_KEY also set

    If only GROQ_API_KEY:
        1. llama-3.3-70b-versatile                   (groq, primary)
        2. meta-llama/llama-4-scout-17b-16e-instruct (groq, secondary)
        3. llama-3.1-8b-instant                       (groq, fallback)

    A failed model enters a 60-second cooldown before being retried.
    """

    def __init__(self):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        openai_key    = os.getenv("OPENAI_API_KEY",    "").strip()
        groq_key      = os.getenv("GROQ_API_KEY",      "").strip()

        if anthropic_key:
            # Anthropic Claude — opus 4.8 is the most capable model for research
            self.primary_config   = ModelConfig("claude-opus-4-8",         provider="anthropic", temperature=0.1, max_tokens=4096)
            self.secondary_config = ModelConfig("claude-sonnet-4-6",       provider="anthropic", temperature=0.1, max_tokens=4096)
            self.fallback_configs = []
            if openai_key:
                self.fallback_configs.append(
                    ModelConfig("gpt-4o-mini", provider="openai", temperature=0.1, max_tokens=4096)
                )
            if groq_key:
                self.fallback_configs.append(
                    ModelConfig("llama-3.3-70b-versatile", provider="groq", temperature=0.1, max_tokens=4096)
                )
        elif openai_key:
            # OpenAI as the primary provider — GPT-4o is best for complex research
            self.primary_config   = ModelConfig("gpt-4o",      provider="openai", temperature=0.1, max_tokens=4096)
            self.secondary_config = ModelConfig("gpt-4o-mini", provider="openai", temperature=0.1, max_tokens=4096)
            self.fallback_configs = []
            if groq_key:
                # Keep Groq as emergency fallbacks when both keys are present
                self.fallback_configs = [
                    ModelConfig("llama-3.3-70b-versatile", provider="groq", temperature=0.1, max_tokens=4096),
                    ModelConfig("llama-3.1-8b-instant",    provider="groq", temperature=0.1, max_tokens=2048),
                ]
        elif groq_key:
            # Groq-only mode — verified live models as of June 2026
            self.primary_config   = ModelConfig("llama-3.3-70b-versatile",                   provider="groq", temperature=0.1, max_tokens=4096)
            self.secondary_config = ModelConfig("meta-llama/llama-4-scout-17b-16e-instruct", provider="groq", temperature=0.1, max_tokens=4096)
            self.fallback_configs = [
                ModelConfig("llama-3.1-8b-instant", provider="groq", temperature=0.1, max_tokens=2048),
            ]
        else:
            raise ValueError("No API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY in .env")

        self.current_config  = self.primary_config
        self.use_secondary   = False
        self.total_requests  = 0
        self.cooldown_period = 60    # seconds before a failed model is retried

    def _is_model_available(self, config: ModelConfig) -> bool:
        """Returns False if the model is within the post-failure cooldown window."""
        if config.last_failure_time is None:
            return True
        return (time.time() - config.last_failure_time) > self.cooldown_period

    def _create_llm_instance(self, config: ModelConfig, temperature: Optional[float] = None):
        """
        Instantiates the right LLM client based on config.provider.
        Returns None on failure and records failure time for cooldown tracking.

        No probe/test call is made here — failures surface at invoke() time
        and are caught by context_aware_llm's fallback cascade in nodes.py.
        """
        try:
            temp = temperature if temperature is not None else config.temperature

            if config.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(
                    model=config.name,
                    temperature=temp,
                    max_tokens=config.max_tokens,
                )
            elif config.provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=config.name,
                    temperature=temp,
                    max_tokens=config.max_tokens,
                )
            else:  # groq
                llm = ChatGroq(
                    model=config.name,
                    temperature=temp,
                    max_tokens=config.max_tokens,
                )

            config.success_count += 1
            logger.info(f"Initialized {config.provider}/{config.name}")
            return llm

        except Exception as e:
            config.failure_count += 1
            config.last_failure_time = time.time()
            logger.warning(f"Failed to initialize {config.provider}/{config.name}: {e}")
            return None

    def get_llm(self, prefer_secondary: bool = False, force_model: Optional[str] = None):
        """
        Returns the best available LLM, walking the fallback chain if needed.

        Order: preferred primary/secondary (respecting cooldown) → fallback_configs
               → if all in cooldown: retry without restriction → raise.
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

        logger.warning("Primary/secondary in cooldown, trying fallback models...")
        for config in self.fallback_configs:
            if self._is_model_available(config):
                llm = self._create_llm_instance(config)
                if llm:
                    self.current_config = config
                    return llm

        # Last resort: ignore cooldown and try everything once more
        logger.error("All models in cooldown — retrying without restriction...")
        for config in [self.primary_config, self.secondary_config] + self.fallback_configs:
            llm = self._create_llm_instance(config)
            if llm:
                self.current_config = config
                return llm

        raise Exception("All models failed. Check API keys and internet connection.")

    def get_primary_llm(self, **kwargs):
        return self.get_llm(prefer_secondary=False, **kwargs)

    def get_secondary_llm(self, **kwargs):
        return self.get_llm(prefer_secondary=True, **kwargs)

    def switch_to_secondary(self) -> None:
        self.use_secondary = True

    def switch_to_primary(self) -> None:
        self.use_secondary = False

    def get_model_for_task(self, task_type: str):
        """
        Selects model + temperature for the given task type.

        For OpenAI: gpt-4o handles all tasks (temp varies by type).
        For Groq: complex tasks get secondary (llama-4-scout), simple tasks get primary.

        Falls back to get_llm() if the preferred model is in cooldown.
        """
        task_model_mapping = {
            "reasoning": self.secondary_config,   # research/analysis: deeper model
            "math":      self.primary_config,
            "analysis":  self.secondary_config,
            "coding":    self.primary_config,
            "creative":  self.primary_config,
            "general":   self.primary_config,
        }

        temperature    = TASK_TEMPERATURES.get(task_type.lower(), 0.1)
        preferred      = task_model_mapping.get(task_type.lower(), self.primary_config)

        if self._is_model_available(preferred):
            llm = self._create_llm_instance(preferred, temperature=temperature)
            if llm:
                self.current_config = preferred
                logger.info(f"Using {preferred.provider}/{preferred.name} for task={task_type} temp={temperature}")
                return llm

        logger.info(f"Preferred model for '{task_type}' unavailable, falling back")
        return self.get_llm()

    def get_current_model_name(self) -> str:
        return self.current_config.name

    def get_provider(self) -> str:
        return self.current_config.provider

    def get_model_stats(self) -> Dict[str, Any]:
        all_configs = [self.primary_config, self.secondary_config] + self.fallback_configs
        return {
            "total_requests":           self.total_requests,
            "current_model":            self.current_config.name,
            "current_provider":         self.current_config.provider,
            "using_secondary_pref":     self.use_secondary,
            "models": {
                c.name: {
                    "provider":       c.provider,
                    "success_count":  c.success_count,
                    "failure_count":  c.failure_count,
                    "last_failure":   c.last_failure_time,
                    "available":      self._is_model_available(c),
                }
                for c in all_configs
            },
        }

    def reset_model_stats(self) -> None:
        for config in [self.primary_config, self.secondary_config] + self.fallback_configs:
            config.success_count    = 0
            config.failure_count    = 0
            config.last_failure_time = None
        self.total_requests = 0
        logger.info("Model statistics reset")
