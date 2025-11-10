from ..agent_module import ModelSpec
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from .base_module import BaseGuardModule


@dataclass
class SealionGuardModule(BaseGuardModule):
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(model_name="aisingapore/Llama-Guard-Delta-200k"))
    max_retries: int = 10
    representative_token_index: int = 0
    representative_tokens: dict = field(default_factory=lambda:
        {
            "safe": "Safe",
            "s": "Sensitive",
            "unsafe": "Harmful",
        }
    )

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_name)

    def get_safeguard_input(self, module_input: dict) -> dict:
        if module_input.get("response") is None:
            task = "prompt_classification"
            prompt = module_input["prompt"]
            messages = [
                {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the human user's request is safe or unsafe.\nHuman user:{prompt}"}
            ]
        else:
            task = "response_classification"
            prompt = module_input["prompt"]
            response = module_input["response"]
            messages = [
                {"role": "user", "content": f"You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the AI assistant's response is safe or unsafe.\nHuman user:{prompt}\nAI assistant:{response}"}
            ]
        return {"messages": messages, "task": task}