from ..agent_module import ModelSpec
from .base_module import BaseGuardModule
from .polyguard_module import PolyGuardModule
from .sealguard_module import SEALGuardModule
from .qwen3guard_module import Qwen3GuardModule
from .shieldgemma_module import ShieldGemmaModule
from .llamaguard_module import LlamaGuardModule, LlamaGuard4Module
from .sealionguard_module import SealionGuardModule, GemmaSealionGuardModule


class AutoSafeguardModule:
    mapping = {
        "google/shieldgemma-2b": ShieldGemmaModule,
        "google/shieldgemma-9b": ShieldGemmaModule,
        "google/shieldgemma-27b": ShieldGemmaModule,
        "meta-llama/Llama-Guard-3-1B": LlamaGuardModule,
        "meta-llama/Llama-Guard-3-8B": LlamaGuardModule,
        "meta-llama/Llama-Guard-4-12B": LlamaGuard4Module,
        "ToxicityPrompts/PolyGuard-Qwen-Smol": PolyGuardModule,
        "ToxicityPrompts/PolyGuard-Qwen": PolyGuardModule,     
        "ToxicityPrompts/PolyGuard-Ministral": PolyGuardModule,
        "MickyMike/SEALGuard-1.5B": SEALGuardModule,           
        "MickyMike/SEALGuard-7B": SEALGuardModule,             
        "Qwen/Qwen3Guard-Gen-8B": Qwen3GuardModule,
        "aisingapore/Gemma-Guard-4B-Delta": GemmaSealionGuardModule,
        "aisingapore/SEA-Guard-V2": SealionGuardModule,
        "aisingapore/Llama-Guard-Delta-100k": SealionGuardModule,
        "aisingapore/Llama-Guard-Delta-200k": SealionGuardModule,
        "aisingapore/Llama-Guard-Delta-300k-rerun": SealionGuardModule,
        "aisingapore/Llama-Guard-Delta-400k": SealionGuardModule,
        "aisingapore/Llama-Guard-Delta-500k": SealionGuardModule,
        "aisingapore/Llama-Guard-Delta-500k-no-Generic": SealionGuardModule,
        "aisingapore/Gemma-Guard-SEALION-27B-Delta": GemmaSealionGuardModule,
    }

    @classmethod
    def from_model_name(cls, model_name: str) -> BaseGuardModule:
        return cls.mapping[model_name](spec=ModelSpec(model_name=model_name))