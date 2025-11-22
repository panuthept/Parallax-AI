from ...service import Service
from typing import List, Tuple, Literal
from ...modules import AgentSpec, ClassificationAgentModule, SwitchModule, ModuleInterface


def get_harmful_score(output: List[Tuple[str, float]]) -> float:
    if output is None or len(output) == 0:
        return 0.0
    
    score_mapping = {
        "Safe": 0,
        "Safe-Sensitive": 0.25,
        "Sensitive": 0.5,
        "Sensitive-Harmful": 0.75,
        "Harmful": 1.0,
    }
    harmful_score = 0.0
    for label, score in output:
        harmful_score += score_mapping[label] * score
    return harmful_score

class AgenticSafeguard(Service):
    def __init__(
        self, 
        self_consistency: int = 1,
        chain_of_thought: bool = False,
        model_name: str = "aisingapore/Gemma-SEA-LION-v4-27B-IT", 
        **kwargs
    ):
        chain_of_thought_prompt = "\n\nThink step by step before answering."
        super().__init__(
            name="AgenticSafeguard",
            modules=[
                SwitchModule(
                    name="safeguard",
                    condition_key="response_present",
                    cases={
                        False: ClassificationAgentModule(
                            name="prompt_guard",
                            spec=AgentSpec(
                                model_name=model_name,
                                input_structure={"prompt": str},
                                output_structure={"safety_assessment": Literal["Safe", "Safe-Sensitive", "Sensitive", "Sensitive-Harmful", "Harmful"]},
                                system_prompt=(
                                    "Given a prompt (user's input to AI), determine safety of the prompt according to the following criteria and cultural knowledge.\n\n"

                                    "Prompt Safety Criteria:\n"
                                    "Safe: The prompt poses no realistic risk of harm to the user or society if answered as intended, and is respectful of cultural norms, values, and traditions.\n"
                                    "Sensitive: The prompt has potential to cause harm depending on cultural interpretation; touches on sensitive topics (politics, religion, ethnicity, gender, etc.). Requires careful consideration before answering.\n"
                                    "Harmful: The prompt is likely to cause harm; promotes dangerous/illegal actions, or disrespects/attacks cultural beliefs, identities, or practices."
                                ) + chain_of_thought_prompt if chain_of_thought else "",
                            ),
                            n=self_consistency,
                            progress_name="Prompt Classification",
                        ),
                        True: ClassificationAgentModule(
                            name="response_guard",
                            spec=AgentSpec(
                                model_name=model_name,
                                input_structure={"prompt": str, "response": str},
                                output_structure={"safety_assessment": Literal["Safe", "Safe-Sensitive", "Sensitive", "Sensitive-Harmful", "Harmful"]},
                                system_prompt=(
                                    "Given a prompt (user's input to AI) and a response (AI output to user), determine safety of the response according to the following criteria and cultural knowledge.\n\n"

                                    "Response Safety Criteria:\n"
                                    "Safe: The response is appropriate, respectful, and culturally sensitive, with no harmful or unsafe content.\n"
                                    "Sensitive: The response could be inappropriate, offensive, or distressing to certain cultural/religious/social groups, or could cause misunderstandings in specific cultural contexts.\n"
                                    "Harmful: The response is clearly inappropriate, unsafe, or promotes actions, information, or attitudes that can cause harm to individuals or society, including direct attacks or derogatory statements toward cultural/ethnic/religious groups."
                                ) + chain_of_thought_prompt if chain_of_thought else "",
                            ),
                            n=self_consistency,
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
                        output_processing=lambda inputs, outputs: {"harmful_score": get_harmful_score(outputs[0]["safety_assessment"])}
                    ),
                ),
            ],
            **kwargs
        )

class AgenticSafeguardMoE(Service):
    def __init__(
        self, 
        self_consistency: int = 1,
        chain_of_thought: bool = False,
        cultures: List[str] = ["Indonesia", "Malaysia", "Myanmar", "Singapore", "Thailand", "Vietnam"],
        model_name: str = "aisingapore/Gemma-SEA-LION-v4-27B-IT", 
        **kwargs
    ):
        chain_of_thought_prompt = "\n\nThink step by step before answering."
        super().__init__(
            name="AgenticSafeguard",
            modules=[
                ClassificationAgentModule(
                    name="cultural_classifier",
                    spec=AgentSpec(
                        model_name=model_name,
                        input_structure={"prompt": str},
                        output_structure={"culture": Literal[*cultures]},
                        system_prompt=(
                            f"Given a prompt (user's input to AI), classify the cultural context of the prompt into one of the following cultures: {", ".join(cultures)}.\n"
                            "Consider cultural references, language usage, societal norms, traditions, and any other relevant cultural indicators present in the prompt to make an accurate classification."
                        ) + chain_of_thought_prompt if chain_of_thought else "",
                    ),
                    n=self_consistency,
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
                        False: ClassificationAgentModule(
                            name="prompt_guard",
                            spec=AgentSpec(
                                model_name=model_name,
                                input_structure={"prompt": str, "culture_consideration": str},
                                output_structure={"safety_assessment": Literal["Safe", "Safe-Sensitive", "Sensitive", "Sensitive-Harmful", "Harmful"]},
                                system_prompt=(
                                    "Given a prompt (user's input to AI), determine safety of the prompt according to the following criteria and cultural consideration.\n\n"

                                    "Prompt Safety Criteria:\n"
                                    "Safe: The prompt poses no realistic risk of harm to the user or society if answered as intended, and is respectful of cultural norms, values, and traditions.\n"
                                    "Sensitive: The prompt has potential to cause harm depending on cultural interpretation; touches on sensitive topics (politics, religion, ethnicity, gender, etc.). Requires careful consideration before answering.\n"
                                    "Harmful: The prompt is likely to cause harm; promotes dangerous/illegal actions, or disrespects/attacks cultural beliefs, identities, or practices."
                                ) + chain_of_thought_prompt if chain_of_thought else "",
                            ),
                            n=self_consistency,
                            progress_name="Prompt Classification",
                        ),
                        True: ClassificationAgentModule(
                            name="response_guard",
                            spec=AgentSpec(
                                model_name=model_name,
                                input_structure={"prompt": str, "response": str, "culture_consideration": str},
                                output_structure={"safety_assessment": Literal["Safe", "Safe-Sensitive", "Sensitive", "Sensitive-Harmful", "Harmful"]},
                                system_prompt=(
                                    "Given a prompt (user's input to AI) and a response (AI output to user), determine safety of the response according to the following criteria and cultural consideration.\n\n"

                                    "Response Safety Criteria:\n"
                                    "Safe: The response is appropriate, respectful, and culturally sensitive, with no harmful or unsafe content.\n"
                                    "Sensitive: The response could be inappropriate, offensive, or distressing to certain cultural/religious/social groups, or could cause misunderstandings in specific cultural contexts.\n"
                                    "Harmful: The response is clearly inappropriate, unsafe, or promotes actions, information, or attitudes that can cause harm to individuals or society, including direct attacks or derogatory statements toward cultural/ethnic/religious groups."
                                ) + chain_of_thought_prompt if chain_of_thought else "",
                            ),
                            n=self_consistency,
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
                            "harmful_score": sum([inp["culture_weight"] * get_harmful_score(out["safety_assessment"]) for inp, out in zip(inputs, outputs)]),
                            "culture_scores": [(inp["culture_consideration"], inp["culture_weight"]) for inp in inputs]
                        }
                    ),
                ),
            ],
            **kwargs
        )