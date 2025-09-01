import json
import jsonschema
from abc import ABC, abstractmethod
from parallax_ai.clients import ParallaxOpenAIClient
from typing import Any, List, Tuple, Optional, Iterator


class BaseAgent(ABC):
    @abstractmethod
    def run(self, inputs: List[Any]) -> List[str]:
        pass

    @abstractmethod
    def irun(self, inputs: List[Any]) -> Iterator[str]:
        pass

    @abstractmethod
    def irun_unordered(self, inputs: List[Any]) -> Iterator[Tuple[int, str]]:
        pass


class Agent(BaseAgent):
    def __init__(
        self, 
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        self.model = model
        self.client = ParallaxOpenAIClient(api_key=api_key, base_url=base_url)
        self.system_prompt = system_prompt

    def _inputs_processing(self, inputs):
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], dict)):
            inputs = [inputs]

        processed_inputs = []
        for input in inputs:
            if self.system_prompt is None:
                processed_inputs.append(input)
            else:
                if isinstance(input, str):
                    processed_inputs.append([{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input}])
                elif isinstance(input, list) and isinstance(input[0], dict):
                    if input[0]["role"] == "system":
                        print("System prompt already exists, use the existing one")
                    else:
                        input.insert(0, {"role": "system", "content": self.system_prompt})
                    processed_inputs.append(input)
                else:
                    raise ValueError(f"Unknown input type:\n{input}")
        return processed_inputs

    def _output_processing(self, input, output) -> str:
        if output is None:
            return None
        if isinstance(input, list) and isinstance(input[0], dict):
            return output.choices[0].message.content
        elif isinstance(input, str):
            return output.choices[0].text
        else:
            raise ValueError(f"Unknown input type:\n{input}")

    def run(
        self, 
        inputs, 
    ) -> List[str]:
        processed_inputs = self._inputs_processing(inputs)
        outputs = self.client.run(
            inputs=processed_inputs,
            model=self.model,
        )
        return [self._output_processing(input, output) for input, output in zip(processed_inputs, outputs)]

    def irun(
        self, 
        inputs, 
    ) -> Iterator[str]:
        processed_inputs = self._inputs_processing(inputs)
        for i, output in enumerate(self.client.irun(
            inputs=processed_inputs,
            model=self.model,
        )):
            yield self._output_processing(processed_inputs[i], output)

    def irun_unordered(
        self, 
        inputs, 
    ) -> Iterator[Tuple[int, str]]:
        processed_inputs = self._inputs_processing(inputs)
        for i, output in self.client.irun_unordered(
            inputs=processed_inputs,
            model=self.model,
        ):
            yield i, self._output_processing(processed_inputs[i], output)


class JSONOutputAgent(Agent):
    """ 
    This class receives a JSON schema and provides an instruction and a method to parse and validate the model's output. 
    """
    def __init__(
        self, 
        model: str,
        output_schema: dict|list[dict],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tries: int = 3,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.max_tries = max_tries
        self.output_schema = output_schema
        self.system_prompt = self.system_prompt + "\n\n" + self.output_schema_instruction if self.system_prompt is not None else self.output_schema_instruction

    @property
    def output_schema_instruction(self) -> str:
        return (
            "Output in JSON format that matches the following schema:\n"
            "{output_schema}"
        ).format(output_schema=json.dumps(self.output_schema, indent=2))

    def json_parser(self, output: str) -> dict:
        try:
            # Remove prefix and suffix texts
            output = output.split("```json")
            if len(output) != 2:
                return None
            output = output[1].split("```")
            if len(output) != 2:
                return None
            output = output[0].strip()

            # Parse the JSON object
            output = json.loads(output)

            # Validate the JSON object
            jsonschema.validate(instance=output, schema=self.output_schema)
            return output
        except Exception:
            return None
        
    def _inputs_processing(self, inputs):
        processed_inputs = super()._inputs_processing(inputs)

        duplicated_processed_inputs = []
        for processed_input in processed_inputs:
            duplicated_processed_inputs.extend(processed_input * self.max_tries)
        return duplicated_processed_inputs

    def _output_processing(self, input, output) -> dict|list[dict]:
        output = super()._output_processing(input, output)
        return self.json_parser(output)

    def run(
        self, 
        inputs, 
    ) -> List[str]:
        duplicated_outputs = super().run(inputs)
        # Take the first valid output from the duplicated outputs
        outputs = []
        for i in range(len(inputs)):
            for j in range(self.max_tries):
                output = duplicated_outputs[i * self.max_tries + j]
                if output is not None:
                    outputs.append(output)
                    break
            if len(outputs) == i:
                # All tries failed, return None
                outputs.append(None)
        assert len(outputs) == len(inputs)
        return outputs

    def irun(
        self, 
        inputs, 
    ) -> Iterator[str]:
        # Take the first valid output, skip the rest
        tracking_index = 0
        for i, output in enumerate(super().irun(inputs)):
            if tracking_index * self.max_tries <= i < (tracking_index + 1) * self.max_tries:
                if output is not None:
                    tracking_index += 1
                    yield output
                else:
                    if i == (tracking_index + 1) * self.max_tries - 1:
                        # All tries failed, return None
                        tracking_index += 1
                        yield None

    def irun_unordered(
        self, 
        inputs, 
    ) -> Iterator[Tuple[int, str]]:
        # Take the first valid output, skip the rest
        tracking_index = 0
        cached_outputs = {i: [] for i in range(len(inputs))}
        for i, output in super().irun_unordered(inputs):
            cached_outputs[i].append(output)
            if i == tracking_index:
                # Get current output
                if output is not None:
                    tracking_index += 1
                    yield i, output
                else:
                    if len(cached_outputs[i]) >= self.max_tries:
                        # All tries failed, return None
                        tracking_index += 1
                        yield i, None


if __name__ == "__main__":
    from time import time

    agent = JSONOutputAgent(
        model="google/gemma-3-27b-it",
        output_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "gender": {"type": "string"},
                },
                "required": ["name", "age", "gender"],
            },
        },
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        system_prompt="Generate a list of 3 persons with name, age, and gender, that relavant to a given user input.",
    )

    messages = [
        {"role": "user", "content": "Thai singers"},
    ]
    
    start_time = time()
    for i, output in enumerate(agent.irun(messages)):
        print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}, Output: {output}")