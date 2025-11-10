from dataclasses import dataclass
from ..agent_module import ModelSpec
from .base_module import BaseGuardModule


@dataclass
class Qwen3GuardModule(BaseGuardModule):
    spec: ModelSpec = ModelSpec(model_name="Qwen/Qwen3Guard-Gen-8B")
    max_retries: int = 10
    representative_token_index: int = 2
    representative_tokens: dict = {
        " Safe": "Safe",
        " Cont": "Sensitive",
        " Unsafe": "Harmful",
    }