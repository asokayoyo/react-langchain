"""Create the AgentCallbackHandler in react-langchain that implements the BaseCallbackHandler"""

from typing import Dict, Any, List
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.callbacks import BaseCallbackHandler


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for the agent"""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Callback handler for the LLM start"""
        print("\n****************")
        print("LLM Start")
        print(f"Prompts: {prompts[0]}")
        print(f"Serialized: {serialized}")
        print("\n****************")

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Callback handler for the LLM end"""
        print("\n****************")
        print("LLM End")
        print(f"Response: {response.generations[0][0].text}")
        print("\n****************")
