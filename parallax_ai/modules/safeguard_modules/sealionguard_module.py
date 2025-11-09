from dataclasses import dataclass
from ..agent_module import ModelSpec
from .base_module import BaseGuardModule


@dataclass
class SealionGuardModule(BaseGuardModule):
    spec: ModelSpec = ModelSpec(model_name="aisingapore/Llama-Guard-Delta-200k")
    max_retries: int = 10
    representative_token_index: int = 0
    representative_tokens = {
        "safe": "Safe",
        "s": "Sensitive",
        "unsafe": "Harmful",
    }

    def get_safeguard_input(self, module_input: dict) -> list:
        if module_input.get("response") is None:
            messages = [
                {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the human user's request is safe or unsafe.\nHuman user:{module_input['prompt']}"}
            ]
        else:
            messages = [
                {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the AI assistant's response is safe or unsafe.\nHuman user:{module_input['prompt']}\nAI assistant:{module_input['response']}"},
            ]
        return {"messages": messages}