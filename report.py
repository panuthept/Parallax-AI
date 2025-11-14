import json
import numpy as np


if __name__ == "__main__":
    base_path = "./benchmark_zeroshot_results/sea_safeguard_bench/cultural_specific"
    splits = ["IN_EN", "MS_EN", "MY_EN", "TA_EN", "TH_EN", "TL_EN", "VI_EN"]
    model_names = {
        # "google/shieldgemma-2b": "ShieldGemma 2B",
        # "google/shieldgemma-9b": "ShieldGemma 9B",
        # "google/shieldgemma-27b": "ShieldGemma 27B",
        # "meta-llama/Llama-Guard-3-1B": "LlamaGuard-3 1B",
        # "meta-llama/Llama-Guard-3-8B": "LlamaGuard-3 8B",
        # "ToxicityPrompts/PolyGuard-Qwen-Smol": "PolyGuard-Qwen 0.5B",
        # "ToxicityPrompts/PolyGuard-Qwen": "PolyGuard-Qwen 8B",
        # "ToxicityPrompts/PolyGuard-Ministral": "PolyGuard-Ministral 8B",
        # "MickyMike/SEALGuard-1.5B": "SEALGuard 1.5B",
        # "MickyMike/SEALGuard-7B": "SEALGuard 7B",
        # "Qwen/Qwen3Guard-Gen-8B": "Qwen3Guard-Gen 8B",
        # "aisingapore/Gemma-Guard-4B-Delta": "SEA-Guard 4B (Our)",
        # "aisingapore/SEA-Guard-V2": "SEA-Guard 8B (Our)",
        # "aisingapore/Llama-Guard-Delta-100k": "SEA-Guard-100K 8B (Our)",
        # "aisingapore/Llama-Guard-Delta-200k": "SEA-Guard-200K 8B (Our)",
        # "aisingapore/Llama-Guard-Delta-300k-rerun": "SEA-Guard-300K 8B (Our)",
        # "aisingapore/Llama-Guard-Delta-400k": "SEA-Guard-400K 8B (Our)",
        # "aisingapore/Llama-Guard-Delta-500k": "SEA-Guard-500K 8B (Our)",
        # "aisingapore/Llama-Guard-Delta-500k-no-Generic": "SEA-Guard-500K-NG 8B (Our)",
        # "aisingapore/Gemma-Guard-SEALION-27B-Delta": "SEA-Guard 27B (Our)",
        # "aisingapore/Gemma-SEA-LION-v4-27B-IT": "SEA-AgenticGuard 27B (Our)",

        "google/gemma-3-27b-it": "Gemma-3-it 27B",
        "aisingapore/Gemma-SEA-LION-v4-27B-IT": "Gemma-SEA-LION-v4-27B",
        # "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-it 70B",
        # "meta-llama/Llama-3.3-70B-Instruct": "Llama-3.3-it 70B",
        # "openai/gpt-oss-20b": "GPT-OSS 20B",
        # "openai/gpt-oss-120b": "GPT-OSS 120B",
    }
    
    for model_name, display_name in model_names.items():
        results = []
        prompt_avg = []
        response_avg = []
        for split in splits:
            prompt_performance = round(json.load(open(f"{base_path}/{split}/{model_name}/prompt_classification.json"))["performance"]["pr_auc"] * 100, 1)
            response_performance = round(json.load(open(f"{base_path}/{split}/{model_name}/response_classification.json"))["performance"]["pr_auc"] * 100, 1)
            prompt_avg.append(prompt_performance)
            response_avg.append(response_performance)
            results.append(f"{prompt_performance} / {response_performance}")
        prompt_avg = round(np.mean(prompt_avg), 1)
        response_avg = round(np.mean(response_avg), 1)
        results.append(f"{prompt_avg} / {response_avg}")
        print("& " + display_name + " & " + " & ".join(results) + " \\\\")
        print("%")