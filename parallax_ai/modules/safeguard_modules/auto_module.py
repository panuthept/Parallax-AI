from .polyguard_module import PolyGuardModule
from .sealguard_module import SEALGuardModule
from .base_class import GuardModule, GuardSpec
from .qwen3guard_module import Qwen3GuardModule
from .shieldgemma_module import ShieldGemmaModule
from .llamaguard_module import LlamaGuardModule, LlamaGuard4Module
from .sealionguard_module import SealionGuardModule, GemmaSealionGuardModule


class AutoSafeguardModule:
    available_safeguards = {
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
        "Qwen/Qwen3Guard-Gen-4B": Qwen3GuardModule,
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
        "aisingapore/1M_SEA-Guard_qwen3-4b_Non_Bias": SealionGuardModule,
        "aisingapore/1M_SEA-Guard_qwen3-8b_Non_Bias": SealionGuardModule,
        "aisingapore/1M_SEA-Guard_llama-8b_Non_Bias": SealionGuardModule,
        "aisingapore/1M_SEA-Guard_gemma3-12b_Non_Bias": GemmaSealionGuardModule,
    }

    @classmethod
    def from_spec(cls, spec: GuardSpec, **kwargs) -> GuardModule:
        if spec.model_name not in cls.available_safeguards:
            raise ValueError(f"No safeguard module found for model name: {spec.model_name}")
        return cls.available_safeguards[spec.model_name](spec=spec, **kwargs)