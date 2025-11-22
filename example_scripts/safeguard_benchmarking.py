import os
import json
import argparse
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
    benchmark_name = "sea_safeguard_bench"
    benchmark = SEASafeguardBench()

    # Run benchmark for cultural specific
    prompt_scores = []
    response_scores = []
    for model_name in worker_nodes.keys():
        safeguard = get_safeguard(args, worker_nodes)
        for split in ["IN_EN", "MS_EN", "MY_EN", "TA_EN", "TH_EN", "TL_EN", "VI_EN"]:
            save_path = f"./outputs/{save_name}/{benchmark_name}/cultural_specific/{split}"

            if not os.path.exists(f"{save_path}/prompt_classification.json"):
                results = benchmark.evaluate(
                    label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
                    subsets=["cultural_content_generation", "cultural_in_the_wild"],
                    splits=[split],
                    task="prompt_classification",
                    safeguard=safeguard,
                    debug_mode=debug_mode,
                    verbose=True,
                )
                print(results["performance"])

                os.makedirs(save_path, exist_ok=True)
                with open(f"{save_path}/prompt_classification.json", "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            else:
                with open(f"{save_path}/prompt_classification.json", "r") as f:
                    results = json.load(f)
                print(results["performance"])
            prompt_scores.append(round(results["performance"]["pr_auc"] * 100, 1))

            if not os.path.exists(f"{save_path}/response_classification.json"):
                results = benchmark.evaluate(
                    label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
                    subsets=["cultural_content_generation"],
                    splits=[split],
                    task="response_classification",
                    safeguard=safeguard,
                    debug_mode=debug_mode,
                    verbose=True,
                )
                print(results["performance"])

                os.makedirs(save_path, exist_ok=True)
                with open(f"{save_path}/response_classification.json", "w") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            else:
                with open(f"{save_path}/response_classification.json", "r") as f:
                    results = json.load(f)
                print(results["performance"])
            response_scores.append(round(results["performance"]["pr_auc"] * 100, 1))

    # Report performance
    prompt_avg_score = round(sum(prompt_scores) / len(prompt_scores), 1)
    response_avg_score = round(sum(response_scores) / len(response_scores), 1)
    prompt_scores.append(prompt_avg_score)
    response_scores.append(response_avg_score)
    print(" & ".join([f"{prompt_score} / {response_score}" for prompt_score, response_score in zip(prompt_scores, response_scores)]))
