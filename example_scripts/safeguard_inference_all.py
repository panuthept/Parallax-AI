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
                save_path = f"./outputs/{model_name}/{benchmark_name}/{subset}/{split}/{language}"
                if not os.path.exists(f"{save_path}/prompt_classification.json"):
                    results = benchmark.evaluate(
                        label_mapping={"Harmful": 1.0, "Sensitive": 0.0, "Safe": 0.0},
                        subsets=[subset],
                        splits=[split],
                        language=language,
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
                
                if not os.path.exists(f"{save_path}/response_classification.json"):
                    results = benchmark.evaluate(
                        label_mapping={"Harmful": 1.0, "Sensitive": 1.0, "Safe": 0.0},
                        subsets=[subset],
                        splits=[split],
                        language=language,
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

    # Report performance summary