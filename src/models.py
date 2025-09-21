from typing import List, Dict, Any, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationState(TypedDict):
    """Enhanced state schema with conversation memory and context"""
    messages: Annotated[List[AnyMessage], add_messages]
    user_context: Dict[str, Any]  # Store user preferences, history, etc.
    tool_results_cache: Dict[str, Any]  # Cache recent tool results
    conversation_summary: str  # Summary of conversation so far
    error_count: int  # Track errors for fallback strategies
    last_tool_used: str  # Track last successful tool
    current_model_used: str  # Track which model was used
    model_switch_count: int  # Track how many times we've switched models

class ModelConfig:
    """Configuration for individual models"""
    def __init__(self, name: str, temperature: float = 0.1, max_tokens: int = 2000, max_retries: int = 2):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0

class EnhancedLLM:
    """Enhanced LLM class with dual model system and intelligent fallback"""
    
    def __init__(self):
        # Updated with currently supported Groq models (as of September 2025)
        self.primary_config = ModelConfig("llama-3.3-70b-versatile", temperature=0.1, max_tokens=2000)
        self.secondary_config = ModelConfig("llama-3.1-70b-versatile", temperature=0.1, max_tokens=2000)
        
        # Fallback models - using currently supported models
        self.fallback_configs = [
            ModelConfig("llama-3.2-90b-text-preview", temperature=0.1, max_tokens=1800),
            ModelConfig("llama-3.1-8b-instant", temperature=0.1, max_tokens=1500),  # Fast lightweight model
            ModelConfig("gemma2-9b-it", temperature=0.1, max_tokens=1500)  # Most reliable fallback
        ]
        
        self.current_config = self.primary_config
        self.use_secondary = False
        self.total_requests = 0
        self.cooldown_period = 60  # seconds to wait before retrying a failed model
        
    def _is_model_available(self, config: ModelConfig) -> bool:
        """Check if a model is available (not in cooldown)"""
        if config.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - config.last_failure_time
        return time_since_failure > self.cooldown_period
    
    def _create_llm_instance(self, config: ModelConfig) -> Optional[ChatGroq]:
        """Create an LLM instance with the given configuration"""
        try:
            llm = ChatGroq(
                model=config.name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            # Test the model with a simple query
            test_response = llm.invoke("Say 'OK' if you're working.")
            if test_response:
                config.success_count += 1
                logger.info(f"Successfully initialized model: {config.name}")
                return llm
                
        except Exception as e:
            config.failure_count += 1
            config.last_failure_time = time.time()
            logger.warning(f"Failed to initialize model {config.name}: {str(e)}")
            
        return None
    
    def get_llm(self, prefer_secondary: bool = False, force_model: Optional[str] = None) -> ChatGroq:
        """Get LLM instance with intelligent model selection"""
        self.total_requests += 1
        
        # If a specific model is forced
        if force_model:
            for config in [self.primary_config, self.secondary_config] + self.fallback_configs:
                if config.name == force_model:
                    llm = self._create_llm_instance(config)
                    if llm:
                        self.current_config = config
                        return llm
                    break
        
        # Determine model priority order
        if prefer_secondary or self.use_secondary:
            primary_order = [self.secondary_config, self.primary_config]
        else:
            primary_order = [self.primary_config, self.secondary_config]
        
        # Try primary and secondary models first
        for config in primary_order:
            if self._is_model_available(config):
                llm = self._create_llm_instance(config)
                if llm:
                    self.current_config = config
                    self.use_secondary = (config == self.secondary_config)
                    return llm
        
        # If both primary and secondary fail, try fallback models
        logger.warning("Both primary and secondary models failed, trying fallback models...")
        
        for config in self.fallback_configs:
            if self._is_model_available(config):
                llm = self._create_llm_instance(config)
                if llm:
                    self.current_config = config
                    logger.info(f"Using fallback model: {config.name}")
                    return llm
        
        # Last resort: try any model without cooldown restrictions
        logger.error("All models failed with cooldown, trying without restrictions...")
        all_configs = [self.primary_config, self.secondary_config] + self.fallback_configs
        
        for config in all_configs:
            llm = self._create_llm_instance(config)
            if llm:
                self.current_config = config
                logger.info(f"Emergency fallback to: {config.name}")
                return llm
        
        raise Exception("All models failed to initialize. Please check your GROQ_API_KEY and internet connection.")
    
    def get_primary_llm(self, **kwargs) -> ChatGroq:
        """Explicitly get primary model"""
        return self.get_llm(prefer_secondary=False, **kwargs)
    
    def get_secondary_llm(self, **kwargs) -> ChatGroq:
        """Explicitly get secondary model"""
        return self.get_llm(prefer_secondary=True, **kwargs)
    
    def switch_to_secondary(self) -> None:
        """Switch default preference to secondary model"""
        self.use_secondary = True
        logger.info("Switched to preferring secondary model")
    
    def switch_to_primary(self) -> None:
        """Switch default preference back to primary model"""
        self.use_secondary = False
        logger.info("Switched to preferring primary model")
    
    def get_model_for_task(self, task_type: str) -> ChatGroq:
        """Get the best model for a specific task type"""
        task_model_mapping = {
            "reasoning": self.secondary_config,  # Llama 3.1 70B for complex reasoning
            "math": self.primary_config,         # Llama 3.3 70B for mathematical tasks
            "analysis": self.secondary_config,   # Llama 3.1 70B for analysis
            "coding": self.primary_config,       # Llama 3.3 70B for coding
            "creative": self.primary_config,     # Llama 3.3 70B for creative tasks
            "general": self.primary_config,      # Llama 3.3 70B for general conversation
        }
        
        preferred_config = task_model_mapping.get(task_type.lower(), self.primary_config)
        
        if self._is_model_available(preferred_config):
            llm = self._create_llm_instance(preferred_config)
            if llm:
                self.current_config = preferred_config
                logger.info(f"Using {preferred_config.name} for {task_type} task")
                return llm
        
        # Fallback to standard selection
        logger.info(f"Preferred model for {task_type} not available, using standard selection")
        return self.get_llm()
    
    def get_current_model_name(self) -> str:
        """Get the name of the currently active model"""
        return self.current_config.name
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about model usage and performance"""
        all_configs = [self.primary_config, self.secondary_config] + self.fallback_configs
        
        stats = {
            "total_requests": self.total_requests,
            "current_model": self.current_config.name,
            "using_secondary_preference": self.use_secondary,
            "models": {}
        }
        
        for config in all_configs:
            stats["models"][config.name] = {
                "success_count": config.success_count,
                "failure_count": config.failure_count,
                "last_failure": config.last_failure_time,
                "available": self._is_model_available(config)
            }
        
        return stats
    
    def reset_model_stats(self) -> None:
        """Reset all model statistics"""
        all_configs = [self.primary_config, self.secondary_config] + self.fallback_configs
        
        for config in all_configs:
            config.success_count = 0
            config.failure_count = 0
            config.last_failure_time = None
        
        self.total_requests = 0
        logger.info("Model statistics reset")

# Helper function for task optimization
def get_task_optimized_llm(llm_manager: EnhancedLLM, user_message: str) -> ChatGroq:
    """Analyze user message and return optimized LLM for the task"""
    message_lower = user_message.lower()
    
    # Task detection patterns
    if any(word in message_lower for word in ['calculate', 'math', 'equation', 'solve', 'compute']):
        return llm_manager.get_model_for_task("math")
    elif any(word in message_lower for word in ['analyze', 'analysis', 'compare', 'evaluate']):
        return llm_manager.get_model_for_task("analysis")
    elif any(word in message_lower for word in ['code', 'programming', 'python', 'javascript', 'debug']):
        return llm_manager.get_model_for_task("coding")
    elif any(word in message_lower for word in ['write', 'story', 'poem', 'creative', 'imagine']):
        return llm_manager.get_model_for_task("creative")
    elif any(word in message_lower for word in ['reason', 'logic', 'think', 'explain', 'why']):
        return llm_manager.get_model_for_task("reasoning")
    else:
        return llm_manager.get_model_for_task("general")