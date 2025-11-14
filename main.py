import os
import json
from parallax_ai.modules import ModelSpec, ModuleInterface
from example_services.agentic_safeguard import AgenticSafeguard, AgenticSafeguardMoE
from parallax_ai.benchmarks import SEALSBench, SEASafeguardBench, PKUSafeRLHFQA
from parallax_ai.modules.safeguards import (
    LlamaGuardModule, 
    LlamaGuard4Module, 
    PolyGuardModule,
    SEALGuardModule,
    ShieldGemmaModule, 
    Qwen3GuardModule, 
    SealionGuardModule,
    GemmaSealionGuardModule,
)


if __name__ == "__main__":
    worker_nodes = {
        # "google/shieldgemma-2b": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.132:8000/v1"},
        # ],
        # "google/shieldgemma-9b": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.137:8000/v1"},
        # ],
        # "google/shieldgemma-27b": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.144:8000/v1"},
        # ],
        # "meta-llama/Llama-Guard-3-1B": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.134:8000/v1"},
        # ],
        # "meta-llama/Llama-Guard-3-8B": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.137:8000/v1"},
        # ],
        "ToxicityPrompts/PolyGuard-Qwen-Smol": [
            {"api_key": "EMPTY", "base_url": f"http://192.168.12.131:8000/v1"},
        ],
        "ToxicityPrompts/PolyGuard-Qwen": [
            {"api_key": "EMPTY", "base_url": f"http://192.168.12.132:8000/v1"},
        ],
        # "ToxicityPrompts/PolyGuard-Ministral": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.144:8000/v1"},
        # ],
        # "meta-llama/Llama-Guard-4-12B": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.135:8000/v1"},
        # ],
        # "MickyMike/SEALGuard-1.5B": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.134:8000/v1"},
        # ],
        # "MickyMike/SEALGuard-7B": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.144:8000/v1"},
        # ],
        # "Qwen/Qwen3Guard-Gen-8B": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.139:8000/v1"},
        # ],

        # "aisingapore/Gemma-Guard-4B-Delta": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.142:8000/v1"},
        # ],
        # "aisingapore/SEA-Guard-V2": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.146:8000/v1"},
        # ],
        # "aisingapore/Llama-Guard-Delta-100k": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.145:8000/v1"},
        # ],
        # "aisingapore/Llama-Guard-Delta-200k": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.141:8000/v1"},
        # ],
        # "aisingapore/Llama-Guard-Delta-300k-rerun": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.143:8000/v1"},
        # ],
        # "aisingapore/Llama-Guard-Delta-400k": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.138:8000/v1"},
        # ],
        # "aisingapore/Llama-Guard-Delta-500k": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.140:8000/v1"},
        # ],
        # "aisingapore/Llama-Guard-Delta-500k-no-Generic": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.135:8000/v1"},
        # ],
        # "aisingapore/Gemma-Guard-SEALION-27B-Delta": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.131:8000/v1"},
        # ],

        # "google/gemma-3-27b-it": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.136:8000/v1"},
        # ],
        # "aisingapore/Gemma-SEA-LION-v4-27B-IT": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.133:8000/v1"},
        # ],
        # "meta-llama/Llama-3.1-70B-Instruct": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.132:8000/v1"},
        # ],
        # "meta-llama/Llama-3.3-70B-Instruct": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.134:8000/v1"},
        # ],
        # "openai/gpt-oss-20b": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.137:8000/v1"},
        # ],
        # "openai/gpt-oss-120b": [
        #     {"api_key": "EMPTY", "base_url": f"http://192.168.12.131:8000/v1"},
        # ],
    }
    safeguards = {
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
        # "google/gemma-3-27b-it": AgenticSafeguardMoE,
        # "aisingapore/Gemma-SEA-LION-v4-27B-IT": AgenticSafeguardMoE,
        # "google/gemma-3-27b-it": AgenticSafeguard,
        # "aisingapore/Gemma-SEA-LION-v4-27B-IT": AgenticSafeguard,
        # "meta-llama/Llama-3.1-70B-Instruct": AgenticSafeguard,
        # "meta-llama/Llama-3.3-70B-Instruct": AgenticSafeguard,
        # "openai/gpt-oss-20b": AgenticSafeguard,
        # "openai/gpt-oss-120b": AgenticSafeguard,
    }
    # benchmark_name = "seals_bench"
    # benchmark = SEALSBench()
    # benchmark_name = "pku_saferlhf_qa"
    # benchmark = PKUSafeRLHFQA()
    benchmark_name = "sea_safeguard_bench"
    benchmark = SEASafeguardBench()
    # subset = "general"
    # languages = ["English"]
    # subset = "cultural_content_generation"
    subset = "cultural_in_the_wild"
    languages = ["English", "Local"]

    for model_name in worker_nodes.keys():
        for split in benchmark.available_subsets_splits[subset]:
            for language in languages:
                # save_path = f"./outputs/safeguards/{model_name}/{benchmark_name}/{subset}/{split}"
                save_path = f"./outputs/safeguards/{model_name}/{benchmark_name}/{subset}/{split}/{language}"

                if not os.path.exists(f"{save_path}/prompt_classification.json"):
                    results = benchmark.run(
                        subsets=[subset],
                        splits=[split],
                        language=language,
                        task="prompt_classification",
                        # safeguard=safeguards[model_name](
                        #     model_name=model_name,
                        #     worker_nodes=worker_nodes,
                        # ),
                        safeguard=safeguards[model_name](
                            name="Safeguard",
                            spec=ModelSpec(model_name=model_name),
                            interface=ModuleInterface(
                                dependencies=["prompt"],
                                output_processing=lambda inputs, outputs: {"harmful_score": outputs[0]["harmful_score"]}
                            ),
                            progress_name="Prompt Classification",
                            worker_nodes=worker_nodes,
                        ),
                        debug_mode=False,
                        verbose=True
                    )
                    # print(results["performance"])

                    os.makedirs(save_path, exist_ok=True)
                    with open(f"{save_path}/prompt_classification.json", "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)

                if subset == "cultural_in_the_wild":
                    continue

                if not os.path.exists(f"{save_path}/response_classification.json"):
                    results = benchmark.run(
                        subsets=[subset],
                        splits=[split],
                        language=language,
                        task="response_classification",
                        # safeguard=safeguards[model_name](
                        #     model_name=model_name,
                        #     worker_nodes=worker_nodes,
                        # ),
                        safeguard=safeguards[model_name](
                            name="Safeguard",
                            spec=ModelSpec(model_name=model_name),
                            interface=ModuleInterface(
                                dependencies=["prompt", "response"],
                                output_processing=lambda inputs, outputs: {"harmful_score": outputs[0]["harmful_score"]}
                            ),
                            progress_name="Response Classification",
                            worker_nodes=worker_nodes,
                        ),
                        debug_mode=False,
                        verbose=True
                    )
                    # print(results["performance"])

                    os.makedirs(save_path, exist_ok=True)
                    with open(f"{save_path}/response_classification.json", "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)