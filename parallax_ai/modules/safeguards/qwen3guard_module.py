from ..agent_module import ModelSpec
from dataclasses import dataclass, field
from .base_module import BaseGuardModule


@dataclass
class Qwen3GuardModule(BaseGuardModule):
    spec: ModelSpec = field(default_factory=lambda: ModelSpec(model_name="Qwen/Qwen3Guard-Gen-8B"))
    max_retries: int = 10
    representative_token_index: int = 2
    representative_tokens: dict = field(default_factory=lambda:
        {
            " Safe": "Safe",
            " Cont": "Sensitive",
            " Unsafe": "Harmful",
        }
    )