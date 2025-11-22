from ...service import Service
from ...modules.safeguard_modules import AutoSafeguardModule
from ...modules import SwitchModule, ModuleInterface, ModelSpec


class SafeguardModel(Service):
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-Guard-3-8B", 
        **kwargs
    ):
        super().__init__(
            name="Safeguard",
            modules=[
                SwitchModule(
                    name="safeguard",
                    condition_key="response_present",
                    cases={
                        False: AutoSafeguardModule(
                            name="prompt_guard",
                            spec=ModelSpec(model_name=model_name),
                            progress_name="Prompt Classification",
                        ),
                        True: AutoSafeguardModule(
                            name="response_guard",
                            spec=ModelSpec(model_name=model_name),
                            progress_name="Response Classification",
                        ),
                    },
                    interface=ModuleInterface(
                        dependencies=["prompt"],
                        input_processing=lambda deps: [{
                            "prompt": deps["prompt"], 
                            "response": deps.get("response", None),
                            "response_present": deps.get("response", None) is not None
                        }],
                        output_processing=lambda inputs, outputs: {"harmful_score": outputs[0]["harmful_score"]}
                    ),
                ),
            ],
            **kwargs
        )