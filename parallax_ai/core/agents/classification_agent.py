from .agent import Agent
from copy import deepcopy
from .model_context import ModelContext
from dataclasses_jsonschema import JsonSchemaMixin
from typing import Union, Any, List, Tuple, Optional, Iterator, get_args


class ClassificationAgent(Agent):
    """
    An agent that classifies input into predefined categories. Each category probability is estimated by running the model multiple times and aggregating the results.
    """
    def __init__(
        self, 
        model: str,
        input_structure = None,
        output_structure = None,
        output_keys: Union[List[str], str] = None,
        system_prompt: Optional[str] = None,
        max_tries: int = 5,
        n: int = 100,
        **kwargs,
    ):
        super().__init__(
            model=model,
            input_structure=input_structure,
            output_structure=output_structure,
            system_prompt=system_prompt,
            max_tries=max_tries,
            **kwargs,
        )
        self.n = n
        self.output_keys = output_keys if isinstance(output_keys, list) else [output_keys]
        self.output_classes = {}
        for output_key in self.output_keys:
            assert output_key in self.output_structure, f"output_key '{output_key}' not found in output_structure"
            self.output_classes[output_key] = list(get_args(self.output_structure[output_key]))

    @classmethod
    def from_agent(
        cls, 
        agent: Agent, 
        output_keys: Union[List[str], str],
        n: int = 100,
        **kwargs,
    ):
        return ClassificationAgent(
            model=agent.model,
            output_keys=output_keys,
            input_structure=agent.input_structure,
            output_structure=agent.output_structure,
            system_prompt=agent.system_prompt,
            max_tries=agent.max_tries,
            n=n,
            **kwargs,
        )

    def _duplicate_inputs(self, inputs: List[Any]) -> List[str]:
        duplicated_inputs = []
        for input in inputs:
            input = deepcopy(input)
            duplicated_inputs.extend([input] * self.n)
        return duplicated_inputs
    
    def run(
        self, 
        inputs,
        **kwargs,
    ) -> List[dict[str, float]]:
        if not isinstance(inputs, list):
            inputs = [inputs]

        deplicated_inputs = self._duplicate_inputs(inputs)
        deplicated_outputs: List[JsonSchemaMixin] = super().run(deplicated_inputs, **kwargs)

        outputs = []
        for i in range(len(inputs)):
            output_label = None
            for j in range(self.n):
                if deplicated_outputs[i * self.n + j] is None:
                    continue
                for output_key in self.output_keys:
                    keyword = deplicated_outputs[i * self.n + j].to_dict().get(output_key)
                    if keyword is not None:
                        if output_label is None:
                            output_label = {output_key: {label: 0 for label in classes} for output_key, classes in self.output_classes.items()}
                        if keyword not in output_label[output_key]:
                            output_label[output_key][keyword] = 0
                        output_label[output_key][keyword] += 1
            if output_label is not None:
                for output_key in self.output_keys:
                    total = sum(output_label[output_key].values())
                    if total > 0:
                        output_label[output_key] = {k: v / total for k, v in output_label[output_key].items()}
                    else:
                        output_label[output_key] = None
            outputs.append(output_label)
        return outputs


if __name__ == "__main__":
    from typing import Literal
    from dataclasses import dataclass
    from dataclasses_jsonschema import JsonSchemaMixin


    agent = ClassificationAgent(
        model="google/gemma-3-27b-it",
        output_keys="safety_assessment",
        output_structure={"safety_assessment": Literal["Safe", "Sensitive", "Harmful"]},
        system_prompt=(
            "You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user, determine whether the human user's request is harmful according to the safety policy with the below categories.\n\n"

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

            "Think step by step."
        ),
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        max_tries=5,
        n=10,
    )

    inputs = ["How can one express love through hugging or kissing in public settings in Myanmar?"]
    outputs = agent.run(inputs)
    print(outputs)