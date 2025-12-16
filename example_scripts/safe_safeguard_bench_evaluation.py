import os
import json
import argparse
import numpy as np
from typing import List, Dict, Callable
from lexical_bias_lens import LexicalBiasExploitator
from parallax_ai.datasets import AutoDataset, Dataset
from parallax_ai.metrics.safeguard_metrics import SafeguardMetrics
from parallax_ai.services.safeguards import SafeguardModel, AgenticSafeguard, AgenticSafeguardMoE, ZeroshotSafeguard, ZeroshotSafeguardMoE


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
    elif args.zeroshot:
        if args.moe:
            safeguard = ZeroshotSafeguardMoE(
                model_name=args.model_name,
                worker_nodes=worker_nodes,
                self_consistency=args.self_consistency,
                chain_of_thought=args.chain_of_thought,
            )
        else:
            safeguard = ZeroshotSafeguard(
                model_name=args.model_name,
                worker_nodes=worker_nodes,
                self_consistency=args.self_consistency,
                chain_of_thought=args.chain_of_thought,
            )
    else:
        safeguard = SafeguardModel(model_name=args.model_name, worker_nodes=worker_nodes)
    return safeguard

def compute_result(paths: List[str], label_mapping: Dict[str, float], condition: Callable = None, metrics: str = "pr_auc") -> float:
    samples = []
    for path in paths:
        data = json.load(open(path, 'r'))
        if condition is not None:
            data = [sample for sample in data if condition(sample)]
        samples.extend(data)
    metrics = SafeguardMetrics(label_mapping=label_mapping)
    score = metrics(
        predicted_scores=[sample["harmful_score"] for sample in samples],
        gold_labels=[sample["gold_harmful_label"] for sample in samples],
        gold_scores=[sample["gold_severity_level"] for sample in samples],
    )[metrics]
    return round(score * 100, 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run safeguard benchmarking.')
    parser.add_argument('--save_name', type=str, default=None, help='Name to save the benchmark results')
    parser.add_argument('--lens_path', type=str, default=None, help='Name to load the lexical bias lens')
    parser.add_argument('--model_name', type=str, required=False, help='Model name to benchmark')
    parser.add_argument('--api_key', type=str, default='EMPTY', help='API key for the model')
    parser.add_argument('--base_url', type=str, default=None, help='Base URL for the model')
    parser.add_argument('--model_address', type=str, default=None, help='Model ip address to benchmark')
    parser.add_argument('--agentic', action='store_true')
    parser.add_argument('--zeroshot', action='store_true')
    parser.add_argument('--moe', action='store_true')
    parser.add_argument('--self_consistency', type=int, default=1)
    parser.add_argument('--chain_of_thought', action='store_true')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--debug_mode', action='store_true')
    parser.add_argument('--report_mode', action='store_true')
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--n', type=int, default=1)
    args = parser.parse_args()

    args.dataset_name = "sea_safeguard_bench"
    args.save_name = args.save_name if args.save_name is not None else args.model_name
    args.base_url = args.base_url if args.model_address is None else f"http://{args.model_address}:8000/v1"

    exploitator = None
    if args.lens_path is not None:
        exploitator = LexicalBiasExploitator.load(args.lens_path, metric="LMI")

    if not args.report_mode:
        worker_nodes = {
            args.model_name: [
                {"api_key": args.api_key, "base_url": args.base_url},
            ],
        }
        safeguard = get_safeguard(args, worker_nodes)

        dataset: Dataset = AutoDataset(args.dataset_name)
        for subset in dataset.available_subsets:
            splits = dataset.available_splits[subset]
            for split in splits:
                print()(f"=== Processing {args.dataset_name} | Subset: {subset} | Split: {split} ===")
                save_path = f"./outputs/{args.save_name}/{args.dataset_name}/{subset}/{split}"
                # Get samples
                samples = dataset.get_samples(
                    subset=subset,
                    splits=[split],
                    max_samples=args.max_samples
                )
                prompt_samples = [sample for sample in samples if sample["metadata"]["task"] == "prompt_classification"]
                response_samples = [sample for sample in samples if sample["metadata"]["task"] == "response_classification"]
                # Prompt Classification
                if not os.path.exists(f"{save_path}/prompt_classification.json"):
                    # Apply permutation if available
                    if exploitator is not None:
                        for sample in prompt_samples:
                            if sample["gold_harmful_label"] in ["Safe", "Sensitive"]:
                                permuted_prompt = exploitator.permute(
                                    [sample["prompt"]],
                                    target_label="Unsafe",
                                    top_k=args.top_k,
                                    n=args.n,
                                )[0]
                                sample["prompt"] = permuted_prompt
                            elif sample["gold_harmful_label"] == "Harmful":
                                permuted_prompt = exploitator.permute(
                                    [sample["prompt"]],
                                    target_label="Safe",
                                    top_k=args.top_k,
                                    n=args.n,
                                )[0]
                                sample["prompt"] = permuted_prompt

                    print(prompt_samples[0]["prompt"])
                    outputs = safeguard.run(
                        inputs=prompt_samples,
                        verbose=True,
                        debug_mode=args.debug_mode,
                    )
                    os.makedirs(save_path, exist_ok=True)
                    with open(f"{save_path}/prompt_classification.json", "w") as f:
                        json.dump(outputs, f, indent=4, ensure_ascii=False)

                if len(response_samples) == 0:
                    continue # No response classification for this split
                    
                # Response Classification
                if not os.path.exists(f"{save_path}/response_classification.json"):
                    outputs = safeguard.run(
                        inputs=response_samples,
                        verbose=True,
                        debug_mode=args.debug_mode,
                    )
                    os.makedirs(save_path, exist_ok=True)
                    with open(f"{save_path}/response_classification.json", "w") as f:
                        json.dump(outputs, f, indent=4, ensure_ascii=False)

    # print("\n=== Performance Summary (PR-AUC) ===")
    # # Report performance summary #1
    # results = []
    # # Prompt Classification Results
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/EN/prompt_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
    # ))
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/IN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/MS/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/MY/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/TA/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/TH/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/TL/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/VI/prompt_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
    # ))
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/IN_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/MS_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/MY_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/TA_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/TH_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/TL_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/VI_EN/prompt_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
    #     condition=lambda sample: sample["metadata"]["language"] == "English",
    # ))
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/IN_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/MS_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/MY_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/TA_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/TH_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/TL_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/VI_EN/prompt_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
    #     condition=lambda sample: sample["metadata"]["language"] == "SEA",
    # ))
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/IN_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/MS_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/MY_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TA_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TH_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TL_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/VI_EN/prompt_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
    #     condition=lambda sample: sample["metadata"]["language"] == "English",
    # ))
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/IN_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/MS_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/MY_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TA_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TH_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TL_EN/prompt_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/VI_EN/prompt_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
    #     condition=lambda sample: sample["metadata"]["language"] == "SEA",
    # ))
    # results.append(round(np.mean(results).item(), 1))
    # response_classification_start_index = len(results)
    # # Response Classification Results
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/EN/response_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
    # ))
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/IN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/MS/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/MY/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/TA/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/TH/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/TL/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/VI/response_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
    # ))
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/IN_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/MS_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/MY_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TA_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TH_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TL_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/VI_EN/response_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
    #     condition=lambda sample: sample["metadata"]["language"] == "English",
    # ))
    # results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/IN_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/MS_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/MY_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TA_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TH_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/TL_EN/response_classification.json",
    #         f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/VI_EN/response_classification.json",
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
    #     condition=lambda sample: sample["metadata"]["language"] == "SEA",
    # ))
    # results.append(round(np.mean(results[response_classification_start_index:]).item(), 1))
    # print(" & ".join(map(str, results)))

    # # Report performance summary #2
    # prompt_results = []
    # response_results = []
    # for subset in ["IN_EN", "MS_EN", "MY_EN", "TA_EN", "TH_EN", "TL_EN", "VI_EN"]:
    #     prompt_results.append(compute_result(
    #         paths=[
    #             f"./outputs/{args.save_name}/{args.dataset_name}/cultural_in_the_wild/{subset}/prompt_classification.json",
    #             f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/{subset}/prompt_classification.json",
    #         ],
    #         label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
    #     ))
    #     response_results.append(compute_result(
    #         paths=[
    #             f"./outputs/{args.save_name}/{args.dataset_name}/cultural_content_generation/{subset}/response_classification.json",
    #         ],
    #         label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
    #     ))
    # prompt_results.append(round(np.mean(prompt_results).item(), 1))
    # response_results.append(round(np.mean(response_results).item(), 1))
    # print(" & ".join([f'{prompt_result} / {response_result}' for prompt_result, response_result in zip(prompt_results, response_results)]))

    # # Report performance summary #3
    # prompt_results = []
    # response_results = []
    # prompt_results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/{subset}/prompt_classification.json"
    #         for subset in ["EN", "IN", "MS", "MY", "TA", "TH", "TL", "VI"]
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
    # ))
    # response_results.append(compute_result(
    #     paths=[
    #         f"./outputs/{args.save_name}/{args.dataset_name}/general/{subset}/response_classification.json"
    #         for subset in ["EN", "IN", "MS", "MY", "TA", "TH", "TL", "VI"]
    #     ],
    #     label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
    # ))
    # # prompt_results.append(round(np.mean(prompt_results).item(), 1))
    # # response_results.append(round(np.mean(response_results).item(), 1))
    # print(" & ".join([f'{prompt_result} / {response_result}' for prompt_result, response_result in zip(prompt_results, response_results)]))