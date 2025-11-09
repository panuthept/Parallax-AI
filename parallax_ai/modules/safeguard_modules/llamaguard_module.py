from dataclasses import dataclass
from ..agent_module import ModelSpec
from .base_module import BaseGuardModule


@dataclass
class LlamaGuardModule(BaseGuardModule):
    spec: ModelSpec = ModelSpec(model_name="meta-llama/Llama-Guard-3-8B")
    max_retries: int = 10
    representative_token_index: int = 1
    representative_tokens: dict = {
        "safe": "Safe", 
        "unsafe": "Harmful"
    }