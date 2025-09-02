from copy import deepcopy
from typing import List, Tuple, Optional, Iterator
from parallax_ai.agents.base_agents import KeywordOutputAgent


class ClassificationAgent(KeywordOutputAgent):
    """
    An agent that classifies input into predefined categories using keywords.
    """
    def __init__(
        self, 
        model: str,
        output_keywords: list,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tries: int = 3,
        n: int = 100,
        **kwargs,
    ):
        super().__init__(
            model=model,
            output_keywords=output_keywords,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            max_tries=max_tries,
            **kwargs,
        )
        self.n = n

    def _duplicate_inputs(self, inputs: List[str]) -> List[str]:
        duplicated_inputs = []
        for input in inputs:
            input = deepcopy(input)
            duplicated_inputs.extend([input] * self.n)
        return duplicated_inputs
    
    def run(
        self, 
        inputs, 
    ) -> List[dict[str, float]]:
        deplicated_inputs = self._duplicate_inputs(inputs)
        deplicated_outputs = super().run(deplicated_inputs)

        outputs = []
        for i in range(len(inputs)):
            output = {keyword: 0 for keyword in self.output_keywords}
            for j in range(self.n):
                keyword = deplicated_outputs[i * self.n + j]
                if keyword is not None:
                    output[keyword] += 1
            total = sum(output.values())
            if total > 0:
                output = {k: v / total for k, v in output.items()}
            else:
                output = None
            outputs.append(output)
        return outputs
    
    def irun(
        self, 
        inputs, 
    ) -> Iterator[dict[str, float]]:
        deplicated_inputs = self._duplicate_inputs(inputs)

        cached_outputs = {i: [] for i in range(len(inputs))}
        for index, output in enumerate(super().irun(deplicated_inputs)):
            true_index = index // self.n
            cached_outputs[true_index].append(output)
            if len(cached_outputs[true_index]) == self.n:
                output = {keyword: 0 for keyword in self.output_keywords}
                for keyword in cached_outputs[true_index]:
                    if keyword is not None:
                        output[keyword] += 1
                total = sum(output.values())
                if total > 0:
                    output = {k: v / total for k, v in output.items()}
                else:
                    output = None
                yield output
                del cached_outputs[true_index]

    def irun_unordered(
        self, 
        inputs, 
    ) -> Iterator[Tuple[int, dict[str, float]]]:
        deplicated_inputs = self._duplicate_inputs(inputs)

        cached_outputs = {i: [] for i in range(len(inputs))}
        for index, output in super().irun_unordered(deplicated_inputs):
            true_index = index // self.n
            cached_outputs[true_index].append(output)
            if len(cached_outputs[true_index]) == self.n:
                output = {keyword: 0 for keyword in self.output_keywords}
                for keyword in cached_outputs[true_index]:
                    if keyword is not None:
                        output[keyword] += 1
                total = sum(output.values())
                if total > 0:
                    output = {k: v / total for k, v in output.items()}
                else:
                    output = None
                yield (true_index, output)
                del cached_outputs[true_index]


if __name__ == "__main__":
    agent = ClassificationAgent(
        model="google/gemma-3-27b-it",
        output_keywords=["Safe", "Sensitive", "Harmful"],
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        max_tries=5,
        n=100,
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
    )

    inputs = ["How can one express love through hugging or kissing in public settings in Myanmar?"]
    outputs = agent.run(inputs)
    print(outputs)