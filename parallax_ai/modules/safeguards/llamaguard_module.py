
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