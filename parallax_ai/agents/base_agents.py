import json
import jsonschema
from copy import deepcopy
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
        max_tries: int = 5,
        **kwargs,
    ):
        self.model = model
        self.max_tries = max_tries
        self.system_prompt = system_prompt
        self.client = ParallaxOpenAIClient(api_key=api_key, base_url=base_url)

    def _inputs_processing(self, inputs):
        # Ensure that inputs is a list
        if inputs is None or isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], dict)):
            inputs = [inputs]
        # Add system prompt (if exists) to the inputs
        processed_inputs = []
        for input in inputs:
            input = deepcopy(input)
            if input is None or self.system_prompt is None:
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
        inputs = self._inputs_processing(inputs)

        finished_outputs = {}
        unfinished_inputs = inputs
        for _ in range(self.max_tries):
            unfinished_indices = []
            outputs = self.client.run(inputs=unfinished_inputs, model=self.model)
            for i, output in enumerate(outputs):
                if unfinished_inputs[i] is None:
                    finished_outputs[i] = None
                else:
                    output = self._output_processing(inputs[i], output)
                    if output is not None:
                        finished_outputs[i] = output
                    else:
                        unfinished_indices.append(i)
            if len(unfinished_indices) == 0:
                break
            unfinished_inputs = [inputs[i] for i in unfinished_indices]
        return [finished_outputs[i] if i in finished_outputs else None for i in range(len(inputs))]

    def irun(
        self, 
        inputs, 
    ) -> Iterator[str]:
        inputs = self._inputs_processing(inputs)

        current_index = 0
        finished_outputs = {}
        unfinished_indices = None
        unfinished_inputs = inputs
        for _ in range(self.max_tries):
            true_index_mapping = deepcopy(unfinished_indices) if unfinished_indices else []
            unfinished_indices = []
            for i, output in enumerate(self.client.irun(inputs=unfinished_inputs, model=self.model)):
                if unfinished_inputs[i] is None:
                    finished_outputs[i] = None
                    # Fetch all outputs in finished_outputs that match the current_index
                    while current_index in finished_outputs:
                        yield output
                        current_index += 1
                else:
                    # Convert to true index
                    if len(true_index_mapping) > 0:
                        i = true_index_mapping[i]
                    # Process output
                    output = self._output_processing(inputs[i], output)
                    # Check output validity
                    if output is not None:
                        # Cache valid outputs
                        finished_outputs[i] = output
                        # Fetch all outputs in finished_outputs that match the current_index
                        while current_index in finished_outputs:
                            yield output
                            current_index += 1
                    else:
                        unfinished_indices.append(i)
            if len(unfinished_indices) == 0:
                break
            unfinished_inputs = [inputs[i] for i in unfinished_indices]
        if current_index < len(inputs):
            for i in range(current_index, len(inputs)):
                yield finished_outputs[i] if i in finished_outputs else None

    def irun_unordered(
        self, 
        inputs, 
    ) -> Iterator[Tuple[int, str]]:
        inputs = self._inputs_processing(inputs)

        unfinished_indices = None
        unfinished_inputs = inputs
        for _ in range(self.max_tries):
            true_index_mapping = deepcopy(unfinished_indices) if unfinished_indices else []
            unfinished_indices = []
            for i, output in self.client.irun_unordered(inputs=unfinished_inputs, model=self.model):
                if unfinished_inputs[i] is None:
                    yield (i, None)
                else:
                    # Convert to true index
                    if len(true_index_mapping) > 0:
                        i = true_index_mapping[i]
                    # Process output
                    output = self._output_processing(inputs[i], output)
                    # Check output validity
                    if output is not None:
                        yield (i, output)
                    else:
                        unfinished_indices.append(i)
            if len(unfinished_indices) == 0:
                break
            unfinished_inputs = [inputs[i] for i in unfinished_indices]
        if len(unfinished_indices) > 0:
            for i in unfinished_indices:
                yield (i, None)

            
class PresetOutputAgent(Agent):
    @property
    def output_schema_instruction(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def output_post_processing(self, output):
        raise NotImplementedError

    @abstractmethod
    def output_validation(self, output):
        raise NotImplementedError

    def _inputs_processing(self, inputs):
        """
        Add output_schema_instruction to the inputs
        - If input is a messages with system prompt, this function will append output_schema_instruction to the system prompt.
        - If input is a messages without system prompt, this function will add output_schema_instruction as a system prompt.
        - If input is a prompt, this function will convert input to message then add output_schema_instruction as a system prompt.
        """
        inputs = super()._inputs_processing(inputs)

        processed_inputs = []
        for input in inputs:
            if input is None:
                processed_inputs.append(None)
                continue
            input = deepcopy(input)
            if isinstance(input, str):
                input = [
                    {"role": "system", "content": self.output_schema_instruction},
                    {"role": "user", "content": input},
                ]
                # input = input + "\n\n" + self.output_schema_instruction
            elif isinstance(input, list) and isinstance(input[0], dict):
                if input[0]["role"] == "system":
                    input[0]["content"] = input[0]["content"] + "\n\n" + self.output_schema_instruction
                else:
                    input.insert(0, {"role": "system", "content": self.output_schema_instruction})
            else:
                raise ValueError(f"Unknown input type:\n{input}")
            processed_inputs.append(input)
        return processed_inputs

    def _output_processing(self, input, output) -> dict|list[dict]:
        output = super()._output_processing(input, output)
        output = self.output_post_processing(output)
        return output if self.output_validation(output) else None


class JSONOutputAgent(PresetOutputAgent):
    """ 
    This class receives a JSON schema and provides an instruction and a method to parse and validate the model's output. 
    """
    def __init__(
        self, 
        model: str,
        output_schema: dict,
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
            max_tries=max_tries,
            **kwargs,
        )
        self.output_schema = output_schema

    @property
    def output_schema_instruction(self) -> str:
        return (
            "Output in JSON format that matches the following schema:\n"
            "{output_schema}"
        ).format(output_schema=json.dumps(self.output_schema))

    def output_post_processing(self, output: str) -> dict:
        if output is None:
            return None
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
            return output
        except Exception:
            return None

    def output_validation(self, output: dict) -> bool:
        if output is None:
            return False
        try:
            # Validate the JSON object
            jsonschema.validate(instance=output, schema=self.output_schema)
            return True
        except Exception:
            return False


class KeywordOutputAgent(PresetOutputAgent):
    """ 
    This class receives a list of valid keywords and provides an instruction and a method to parse and validate the model's output. 
    """
    def __init__(
        self, 
        model: str,
        output_keywords: list,
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
            max_tries=max_tries,
            **kwargs,
        )
        self.output_keywords = output_keywords
        
    @property
    def output_schema_instruction(self) -> str:
        return (
            "Finish the output with a single keyword from the following list:\n"
            "{output_keywords}"
        ).format(output_keywords=self.output_keywords)

    def output_post_processing(self, output: str) -> str:
        return output.split()[-1].strip()

    def output_validation(self, output: str) -> bool:
        if output is None:
            return False
        for keyword in self.output_keywords:
            if output.endswith(keyword):
                return True
        return False


if __name__ == "__main__":
    from time import time
    from random import randint

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
        max_tries=5,
    )

    second_agent = KeywordOutputAgent(
        model="google/gemma-3-27b-it",
        output_keywords=["Women", "Men"],
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        max_tries=5,
        system_prompt="Given a person name, determine whether it is men or women name. Think step by step.",
    )

    inputs = [f"Generate a list of {randint(3, 20)} Thai singers" for _ in range(1000)]
    
    # start_time = time()
    # error_count = 0
    # for i, output in enumerate(agent.run(inputs)):
    #     print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}s\nInput: {inputs[i]}\nOutput: {output}")
    #     if output is None:
    #         error_count += 1
    # print(f"Error: {error_count}")
    # print()
    
    # prev_time = None
    # start_time = time()
    # error_count = 0
    # max_iteration_time = 0
    # for i, output in enumerate(agent.irun(inputs)):
    #     iteration_time = 0
    #     if prev_time:
    #         iteration_time = time() - prev_time
    #         if iteration_time > max_iteration_time:
    #             max_iteration_time = iteration_time
    #     if i == 0:
    #         first_output_time = time() - start_time
    #     print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}s ({iteration_time:.4f}s)\nInput: {inputs[i]}\nOutput: {output}")
    #     if output is None:
    #         error_count += 1
    #     prev_time = time()
    # print(f"Error: {error_count}")
    # print(f"First Output Time: {first_output_time:4f}")
    # print(f"Max Iteration Time: {max_iteration_time:4f}")
    # print()

    prev_time = time()
    start_time = time()
    error_count = 0
    max_iteration_time = 0
    for i, output in agent.irun_unordered(inputs):
        iteration_time = time() - prev_time
        prev_time = time()
        if iteration_time > max_iteration_time:
            max_iteration_time = iteration_time
        if i == 0:
            first_output_time = time() - start_time
        print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}s ({iteration_time:.4f}s)\nInput: {inputs[i]}\nOutput: {output}")
        if output is None:
            error_count += 1
        else:
            names = [person["name"] for person in output]
            genders = second_agent.run(names)
            print([(person["name"], gender) for person, gender in zip(output, genders)])
    print(f"Error: {error_count}")
    print(f"First Output Time: {first_output_time:4f}")
    print(f"Max Iteration Time: {max_iteration_time:4f}")
    print()