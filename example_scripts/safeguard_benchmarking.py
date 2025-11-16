import os
import json
from parallax_ai.benchmarks import SEASafeguardBench
from parallax_ai.modules import ModelSpec, ModuleInterface
from parallax_ai.modules.safeguards import SealionGuardModule


if __name__ == "__main__":
    worker_nodes = {
        "<model_hf_repo>": [
            {"api_key": "EMPTY", "base_url": f"http://<model_address>:8000/v1"},
        ],
    }
    benchmark_name = "sea_safeguard_bench"
    benchmark = SEASafeguardBench()
    subset = "general"
    languages = ["English"]

    for model_name in worker_nodes.keys():
        for split in ["TA_EN", "TH_EN", "TL_EN", "MS_EN", "IN_EN", "MY_EN", "VI_EN"]:
            for language in languages:
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