
from ..agent_module import ModelSpec
from dataclasses import dataclass, field
from .base_module import BaseGuardModule


@dataclass
class LlamaGuardModule(BaseGuardModule):
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(model_name="meta-llama/Llama-Guard-3-8B"))
    max_retries: int = 10
    representative_token_index: int = 1
    representative_tokens: dict = field(default_factory=lambda: 
        {
            "safe": "Safe", 
            "unsafe": "Harmful"
        }
    )

@dataclass
class LlamaGuard4Module(BaseGuardModule):
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(model_name="meta-llama/Llama-Guard-4-12B"))
    max_retries: int = 10
    representative_token_index: int = 1
    representative_tokens: dict = field(default_factory=lambda: 
        {
            "safe": "Safe", 
            "unsafe": "Harmful"
        }
    )

    def get_safeguard_input(self, module_input: dict) -> dict:
        task = "prompt_classification"
        messages = [{"role": "user", "content": [{"type": "text", "text": module_input["prompt"]}]}] 
        if module_input.get("response") is not None:
            task = "response_classification"
            messages.append({"role": "assistant", "content": [{"type": "text", "text": module_input["response"]}]})
        return {"messages": messages, "task": task}