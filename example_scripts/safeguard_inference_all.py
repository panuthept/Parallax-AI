import os
import json
import argparse
import numpy as np
from typing import List
from parallax_ai.benchmarks import SEASafeguardBench
from parallax_ai.builtins.safeguards import SafeguardModel, AgenticSafeguard, AgenticSafeguardMoE


def get_safeguard(args, worker_nodes):
    if args.agentic:
        if args.moe:
            safeguard = AgenticSafeguardMoE(
                model_name=args.model_name,
                worker_nodes=worker_nodes,
                self_consistency=args.self_consistency,
                chain_of_thought=args.chain_of_thought,
            )
        else:
            safeguard = AgenticSafeguard(
                model_name=args.model_name,
                worker_nodes=worker_nodes,
                self_consistency=args.self_consistency,
                chain_of_thought=args.chain_of_thought,
            )
    else:
        safeguard = SafeguardModel(model_name=args.model_name, worker_nodes=worker_nodes)
    return safeguard

def get_result(paths: List[str], metrics: str = "pr_auc") -> float:
    scores = []
    for path in paths:
        data = json.load(open(path, 'r'))
        score = data['performance'][metrics]
        scores.append(score)
    return round(sum(scores) / len(scores) * 100, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run safeguard benchmarking.')
    parser.add_argument('--save_name', type=str, default=None, help='Name to save the benchmark results')
    parser.add_argument('--model_name', type=str, required=False, help='Model name to benchmark')
    parser.add_argument('--api_key', type=str, default='EMPTY', help='API key for the model')
    parser.add_argument('--base_url', type=str, default=None, help='Base URL for the model')
    parser.add_argument('--model_address', type=str, default=None, help='Model ip address to benchmark')
    parser.add_argument('--agentic', action='store_true')
    parser.add_argument('--moe', action='store_true')
    parser.add_argument('--self_consistency', type=int, default=1)
    parser.add_argument('--chain_of_thought', action='store_true')
    parser.add_argument('--debug_mode', action='store_true')
    args = parser.parse_args()

    save_name = args.save_name if args.save_name is not None else args.model_name
    model_name = args.model_name
    model_address = args.model_address
    api_key = args.api_key
    base_url = args.base_url if args.model_address is None else f"http://{model_address}:8000/v1"
    debug_mode = args.debug_mode

    worker_nodes = {
        model_name: [
            {"api_key": args.api_key, "base_url": base_url},
        ],
    }
    safeguard = get_safeguard(args, worker_nodes)

    # Run benchmark for all subsets and splits
    benchmark_name = "sea_safeguard_bench"
    benchmark = SEASafeguardBench()
    for subset in benchmark.available_subsets_splits.keys():
        print(f"Subset: {subset}")
        languages = ["English", "Local"] if subset != "general" else [None]
        for split in benchmark.available_subsets_splits[subset]:
            print(f"Split: {split}")
            for language in languages:
                print(f"Language: {language}")
                save_path = f"./outputs/{save_name}/{benchmark_name}/{subset}/{split}/{language}"
                if not os.path.exists(f"{save_path}/prompt_classification.json"):
                    results = benchmark.evaluate(
                        label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
                        subsets=[subset],
                        splits=[split],
                        language=language,
                        task="prompt_classification",
                        safeguard=safeguard,
                        debug_mode=debug_mode,
                        verbose=True,
                    )
                    print(results["performance"])
                    os.makedirs(save_path, exist_ok=True)
                    with open(f"{save_path}/prompt_classification.json", "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)
                
                if subset == "cultural_in_the_wild":
                    continue # No response classification for this subset

                if not os.path.exists(f"{save_path}/response_classification.json"):
                    results = benchmark.evaluate(
                        label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
                        subsets=[subset],
                        splits=[split],
                        language=language,
                        task="response_classification",
                        safeguard=safeguard,
                        debug_mode=debug_mode,
                        verbose=True,
                    )
                    print(results["performance"])

                    os.makedirs(save_path, exist_ok=True)
                    with open(f"{save_path}/response_classification.json", "w") as f:
                        json.dump(results, f, indent=4, ensure_ascii=False)

    # Report performance summary
    results = []
    print("\n=== Performance Summary (PR-AUC) ===")
    # Prompt Classification Results
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/general/EN/None/prompt_classification.json",
        ]
    ))
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/general/IN/None/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/MS/None/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/MY/None/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/TA/None/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/TH/None/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/TL/None/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/VI/None/prompt_classification.json",
        ]
    ))
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/IN_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/MS_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/MY_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/TA_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/TH_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/TL_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/VI_EN/English/prompt_classification.json",
        ]
    ))
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/IN_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/MS_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/MY_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/TA_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/TH_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/TL_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_in_the_wild/VI_EN/Local/prompt_classification.json",
        ]
    ))
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/IN_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/MS_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/MY_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TA_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TH_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TL_EN/English/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/VI_EN/English/prompt_classification.json",
        ]
    ))
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/IN_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/MS_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/MY_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TA_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TH_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TL_EN/Local/prompt_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/VI_EN/Local/prompt_classification.json",
        ]
    ))
    results.append(round(np.mean(results).item(), 1))
    response_classification_start_index = len(results)
    # Response Classification Results
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/general/EN/None/response_classification.json",
        ]
    ))
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/general/IN/None/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/MS/None/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/MY/None/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/TA/None/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/TH/None/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/TL/None/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/general/VI/None/response_classification.json",
        ]
    ))
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/IN_EN/English/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/MS_EN/English/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/MY_EN/English/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TA_EN/English/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TH_EN/English/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TL_EN/English/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/VI_EN/English/response_classification.json",
        ]
    ))
    results.append(get_result(
        paths=[
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/IN_EN/Local/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/MS_EN/Local/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/MY_EN/Local/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TA_EN/Local/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TH_EN/Local/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/TL_EN/Local/response_classification.json",
            f"./outputs/{save_name}/{benchmark_name}/cultural_content_generation/VI_EN/Local/response_classification.json",
        ]
    ))
    results.append(round(np.mean(results[response_classification_start_index:]).item(), 1))

    print(" & ".join(map(str, results)))