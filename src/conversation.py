from typing import List
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

class ConversationManager:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        
    def summarize_conversation(self, messages: List[AnyMessage]) -> str:
        """Create a summary of the conversation"""
        if len(messages) < 4:
            return "New conversation"
        
        # Simple summarization - count topics discussed
        topics = set()
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if any(word in content for word in ['weather', 'temperature', 'climate']):
                    topics.add('weather')
                if any(word in content for word in ['calculate', 'math', 'equation']):
                    topics.add('calculations')
                if any(word in content for word in ['code', 'programming', 'python']):
                    topics.add('programming')
                if any(word in content for word in ['research', 'paper', 'study']):
                    topics.add('research')
        
        if topics:
            return f"Discussion topics: {', '.join(topics)}"
        return f"Conversation with {len(messages)//2} exchanges"
    
    def trim_history(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """Trim conversation history to manageable size"""
        if len(messages) <= self.max_history:
            return messages
        
        # Keep system message if present, plus recent messages
        system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
        recent_msgs = messages[-self.max_history:]
        
        return system_msgs + recent_msgs