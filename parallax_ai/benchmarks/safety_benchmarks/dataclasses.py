from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SafetySample:
    messages: List[dict]
    harmful_score: Optional[float] = None
    gold_harmful_score: Optional[float] = None
    gold_harmful_category: Optional[str] = None

    @staticmethod
    def system_prompt(self) -> str:
        for message in self.messages:
            if message["role"] == "system":
                return message["content"]
        return None

    @staticmethod
    def user_prompt(self) -> str:
        for message in self.messages:
            if message["role"] == "user":
                return message["content"]
        return None
    
    @staticmethod
    def assistant_response(self) -> str:
        for message in self.messages:
            if message["role"] == "assistant":
                return message["content"]
        return None