import os
import json
import argparse
from parallax_ai.benchmarks import SEASafeguardBench
from parallax_ai.modules import ModelSpec, ModuleInterface
from parallax_ai.modules.safeguards import SealionGuardModule


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run safeguard benchmarking.')
    parser.add_argument('--model_name', type=str, required=False, help='Model name to benchmark')
    parser.add_argument('--model_address', type=str, required=False, help='Model address to benchmark')
    args = parser.parse_args()

    model_name = args.model_name
    model_address = args.model_address

    worker_nodes = {
        model_name: [
            {"api_key": "EMPTY", "base_url": f"http://{model_address}:8000/v1"},
        ],
    }
    benchmark_name = "sea_safeguard_bench"
    benchmark = SEASafeguardBench()

    # Run benchmark for cultural specific
    prompt_scores = []
    response_scores = []
    for model_name in worker_nodes.keys():
        for split in ["IN_EN", "MS_EN", "MY_EN", "TA_EN", "TH_EN", "TL_EN", "VI_EN"]:
            save_path = f"./outputs/{model_name}/{benchmark_name}/cultural_specific/{split}"

            if not os.path.exists(f"{save_path}/prompt_classification.json"):
                results = benchmark.evaluate(
                    label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
                    subsets=["cultural_content_generation", "cultural_in_the_wild"],
                    splits=[split],
                    task="prompt_classification",
                    safeguard=SealionGuardModule(
                        name="Prompt Safeguard",
                        spec=ModelSpec(model_name=model_name),
                        interface=ModuleInterface(
                            dependencies=["prompt"],
                            output_processing=lambda inputs, outputs: {"harmful_score": outputs[0]["harmful_score"]}
                        ),
                        progress_name="Prompt Classification",
                        worker_nodes=worker_nodes,
                    ),
                    debug_mode=False,
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
                    safeguard=SealionGuardModule(
                        name="Response Safeguard",
                        spec=ModelSpec(model_name=model_name),
                        interface=ModuleInterface(
                            dependencies=["prompt", "response"],
                            output_processing=lambda inputs, outputs: {"harmful_score": outputs[0]["harmful_score"]}
                        ),
                        progress_name="Response Classification",
                        worker_nodes=worker_nodes,
                    ),
                    debug_mode=False,
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
