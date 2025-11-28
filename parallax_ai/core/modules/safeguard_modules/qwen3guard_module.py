from dataclasses import dataclass, field
from .base_class import GuardModule, GuardSpec


@dataclass
class Qwen3GuardModule(GuardModule):
    spec: GuardSpec = field(default_factory=lambda: GuardSpec(model_name="Qwen/Qwen3Guard-Gen-8B"))
    max_retries: int = 10
    representative_token_index: int = 2
    representative_tokens: dict = field(default_factory=lambda:
        {
            " Safe": "Safe",
            " Cont": "Sensitive",
            " Unsafe": "Harmful",
        }
    )