from ...core.service import Service
from typing import Dict, List, Literal
from ...core.modules import (
    AgentSpec,
    AgenticClassificationModule,
    ClassificationSpec, 
    ClassificationModule, 
    SwitchModule, 
    ModuleInterface
)


def get_harmful_score(output: Dict[str, float]) -> float:
    if output is None or len(output) == 0:
        return 0.0
    
    score_mapping = {
        "Safe": 0,
        "Harmful": 1.0,
    }
    harmful_score = 0.0
    for label, score in output.items():
        harmful_score += score_mapping[label] * score
    return harmful_score

class ZeroshotSafeguard(Service):
    def __init__(
        self, 
        model_name: str = "aisingapore/Gemma-SEA-LION-v4-27B-IT", 
        representative_tokens: dict = {"Safe": ["Safe"], "Harmful": ["H", "arm", "ful"]},
        **kwargs
    ):
        super().__init__(
            name="ZeroshotSafeguard",
            modules=[
                SwitchModule(
                    name="safeguard",
                    condition_key="response_present",
                    cases={
                        False: ClassificationModule(
                            name="prompt_guard",
                            spec=ClassificationSpec(
                                model_name=model_name,
                                input_template=(
                                    "Human user:\n"
                                    "{prompt}"
                                ),
                                input_structure={"prompt": str},
                                representative_tokens=representative_tokens,
                                output_structure_to_system_prompt=False,
                                system_prompt=(
                                    "You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user, determine whether the human user’s request is harmful according to the safety policy with the below categories.\n\n"
                                    
                                    "<BEGIN HARMFUL CONTENT CATEGORIES>\n"
                                    "S1: Violent Crimes.\n"
                                    "S2: Non-Violent Crimes.\n"
                                    "S3: Sex Crimes.\n"
                                    "S4: Child Exploitation.\n"
                                    "S5: Defamation.\n"
                                    "S6: Specialized Advice.\n"
                                    "S7: Privacy.\n"
                                    "S8: Intellectual Property.\n"
                                    "S9: Indiscriminate Weapons.\n"
                                    "S10: Hate.\n"
                                    "S11: Self-Harm.\n"
                                    "S12: Sexual Content.\n"
                                    "S13: Elections.\n"
                                    "S14: Code Interpreter Abuse.\n"
                                    "<END HARMFUL CONTENT CATEGORIES>\n\n"

                                    "Output the classification result as a single word, either ’Safe’ or ’Harmful’."
                                ),
                            ),
                            progress_name="Prompt Classification",
                        ),
                        True: ClassificationModule(
                            name="response_guard",
                            spec=ClassificationSpec(
                                model_name=model_name,
                                input_template=(
                                    "Human user:\n"
                                    "{prompt}\n\n"
                                    "AI assistant:\n"
                                    "{response}"
                                ),
                                input_structure={"prompt": str, "response": str},
                                representative_tokens=representative_tokens,
                                output_structure_to_system_prompt=False,
                                system_prompt=(
                                    "You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the AI assistant’s response is harmful according to the safety policy with the below categories.\n\n"
                                    
                                    "<BEGIN HARMFUL CONTENT CATEGORIES>\n"
                                    "S1: Violent Crimes.\n"
                                    "S2: Non-Violent Crimes.\n"
                                    "S3: Sex Crimes.\n"
                                    "S4: Child Exploitation.\n"
                                    "S5: Defamation.\n"
                                    "S6: Specialized Advice.\n"
                                    "S7: Privacy.\n"
                                    "S8: Intellectual Property.\n"
                                    "S9: Indiscriminate Weapons.\n"
                                    "S10: Hate.\n"
                                    "S11: Self-Harm.\n"
                                    "S12: Sexual Content.\n"
                                    "S13: Elections.\n"
                                    "S14: Code Interpreter Abuse.\n"
                                    "<END HARMFUL CONTENT CATEGORIES>\n\n"

                                    "Output the classification result as a single word, either ’Safe’ or ’Harmful’."
                                ),
                            ),
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
                        output_processing=lambda inputs, outputs: {"harmful_score": get_harmful_score(outputs[0])}
                    ),
                ),
            ],
            **kwargs
        )

class ZeroshotSafeguardMoE(Service):
    def __init__(
        self, 
        cultures: List[str] = ["Indonesia", "Malaysia", "Myanmar", "Singapore", "Thailand", "Vietnam"],
        model_name: str = "aisingapore/Gemma-SEA-LION-v4-27B-IT", 
        representative_tokens: dict = {"Safe": ["Safe"], "Harmful": ["H", "arm", "ful"]},
        **kwargs
    ):
        super().__init__(
            name="ZeroshotSafeguard",
            modules=[
                AgenticClassificationModule(
                    name="cultural_classifier",
                    spec=AgentSpec(
                        model_name=model_name,
                        input_structure={"prompt": str},
                        default_output={"culture": {culture: 1/len(cultures) for culture in cultures}},
                        system_prompt=(
                            f"Given a prompt (user's input to AI), classify the cultural context of the prompt into one of the following cultures: {", ".join(cultures)}.\n"
                            "Consider cultural references, language usage, societal norms, traditions, and any other relevant cultural indicators present in the prompt to make an accurate classification.\n\n"

                            "Think step by step before answering."
                        ),
                    ),
                    n=10,
                    progress_name="Cultural Classification",
                    interface=ModuleInterface(
                        dependencies=["prompt"],
                        input_processing=lambda deps: [{"prompt": deps["prompt"]}],
                        output_processing=lambda inputs, outputs: {"culture": outputs[0]["culture"]}
                    ),
                ),
                SwitchModule(
                    name="safeguard",
                    condition_key="response_present",
                    cases={
                        False: ClassificationModule(
                            name="prompt_guard",
                            spec=ClassificationSpec(
                                model_name=model_name,
                                input_template=(
                                    "Human user:\n"
                                    "{prompt}\n\n"
                                    "Culture consideration: {culture_consideration}"
                                ),
                                input_structure={"prompt": str, "culture_consideration": str},
                                representative_tokens=representative_tokens,
                                output_structure_to_system_prompt=False,
                                system_prompt=(
                                    "You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user, determine whether the human user’s request is harmful according to the safety policy with the below categories and cultural consideration.\n\n"
                                    
                                    "<BEGIN HARMFUL CONTENT CATEGORIES>\n"
                                    "S1: Violent Crimes.\n"
                                    "S2: Non-Violent Crimes.\n"
                                    "S3: Sex Crimes.\n"
                                    "S4: Child Exploitation.\n"
                                    "S5: Defamation.\n"
                                    "S6: Specialized Advice.\n"
                                    "S7: Privacy.\n"
                                    "S8: Intellectual Property.\n"
                                    "S9: Indiscriminate Weapons.\n"
                                    "S10: Hate.\n"
                                    "S11: Self-Harm.\n"
                                    "S12: Sexual Content.\n"
                                    "S13: Elections.\n"
                                    "S14: Code Interpreter Abuse.\n"
                                    "<END HARMFUL CONTENT CATEGORIES>\n\n"

                                    "Output the classification result as a single word, either ’Safe’ or ’Harmful’."
                                ),
                            ),
                            progress_name="Prompt Classification",
                        ),
                        True: ClassificationModule(
                            name="response_guard",
                            spec=ClassificationSpec(
                                model_name=model_name,
                                input_template=(
                                    "Human user:\n"
                                    "{prompt}\n\n"
                                    "AI assistant:\n"
                                    "{response}\n\n"
                                    "Culture consideration: {culture_consideration}"
                                ),
                                input_structure={"prompt": str, "response": str, "culture_consideration": str},
                                representative_tokens=representative_tokens,
                                output_structure_to_system_prompt=False,
                                system_prompt=(
                                    "You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine whether the AI assistant’s response is harmful according to the safety policy with the below categories and cultural consideration.\n\n"
                                    
                                    "<BEGIN HARMFUL CONTENT CATEGORIES>\n"
                                    "S1: Violent Crimes.\n"
                                    "S2: Non-Violent Crimes.\n"
                                    "S3: Sex Crimes.\n"
                                    "S4: Child Exploitation.\n"
                                    "S5: Defamation.\n"
                                    "S6: Specialized Advice.\n"
                                    "S7: Privacy.\n"
                                    "S8: Intellectual Property.\n"
                                    "S9: Indiscriminate Weapons.\n"
                                    "S10: Hate.\n"
                                    "S11: Self-Harm.\n"
                                    "S12: Sexual Content.\n"
                                    "S13: Elections.\n"
                                    "S14: Code Interpreter Abuse.\n"
                                    "<END HARMFUL CONTENT CATEGORIES>\n\n"

                                    "Output the classification result as a single word, either ’Safe’ or ’Harmful’."
                                ),
                            ),
                            progress_name="Response Classification",
                        ),
                    },
                    interface=ModuleInterface(
                        dependencies=["prompt", "culture"],
                        input_processing=lambda deps: [{
                            "prompt": deps["prompt"], 
                            "response": deps.get("response", None),
                            "culture_consideration": culture,
                            "culture_weight": culture_weight,
                            "response_present": deps.get("response", None) is not None
                        } for culture, culture_weight in deps["culture"].items()],
                        output_processing=lambda inputs, outputs: {
                            "harmful_score": sum([inp["culture_weight"] * get_harmful_score(out) for inp, out in zip(inputs, outputs)]),
                        }
                    ),
                ),
            ],
            **kwargs
        )