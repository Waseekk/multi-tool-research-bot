"""
src/conversation.py
===================
Stateless helper that runs at the start of every graph turn (conversation_manager node
in app.py). It trims the message list so it doesn't grow unbounded, and builds the
rolling conversation summary that gets injected into the system prompt.

No LLM calls are made here — both operations are pure Python for speed.
"""

from typing import List
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage


class ConversationManager:
    """
    Handles two housekeeping tasks between turns:
      1. trim_history   — keeps the message list at or below max_history entries
      2. summarize_conversation — extracts the last N user questions as a short summary

    The summary is stored in ConversationState.conversation_summary (models.py) and
    read by context_aware_llm (nodes.py) when rebuilding the system prompt each turn.

    To change how many messages are kept: adjust max_history in __init__.
    To change what goes into the summary: edit summarize_conversation.
    """

    def __init__(self, max_history: int = 20):
        # 20 messages (~10 full turns) keeps enough context for follow-up questions
        # while staying well within the model's input limit.
        self.max_history = max_history

    def summarize_conversation(self, messages: List[AnyMessage]) -> str:
        """
        Builds a one-line rolling summary of the last 3 user questions.

        Output format: "question 1 | question 2 | question 3"
        Each question is truncated to 120 characters to keep the system prompt short.

        We use actual user questions (not keywords) because the LLM needs real context
        to understand follow-up queries like "find more papers on that topic".

        Parameters
        ----------
        messages : current message list (may include SystemMessage, HumanMessage, AIMessage, ToolMessage)

        Returns
        -------
        str  e.g. "What is quantum computing? | Find papers on LLMs | Calculate 15% of 2500"
             or   "New conversation" if no human messages exist yet
        """
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        if not human_messages:
            return "New conversation"

        # Last 3 questions — enough context for follow-ups without bloating the prompt
        recent = [m.content[:120] for m in human_messages[-3:]]
        if len(recent) == 1:
            return f"User asked: {recent[0]}"
        return " | ".join(recent)

    def trim_history(self, messages: List[AnyMessage]) -> List[AnyMessage]:
        """
        Trims the message list to max_history entries, keeping SystemMessages intact.

        SystemMessages must always be preserved — dropping the system message mid-
        conversation would strip the LLM's tool instructions on the next turn.
        In practice the system message is rebuilt from scratch each turn (nodes.py),
        so it will be re-injected even if lost here, but we keep it to be safe.

        Parameters
        ----------
        messages : full accumulated message list from MemorySaver checkpoint

        Returns
        -------
        List[AnyMessage]  trimmed list, always starting with any SystemMessages
        """
        if len(messages) <= self.max_history:
            return messages

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        recent_msgs = messages[-self.max_history:]
        return system_msgs + recent_msgs
