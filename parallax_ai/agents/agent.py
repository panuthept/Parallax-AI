import json
from copy import deepcopy
from typing import List, Tuple, Optional, Iterator
from dataclasses_jsonschema import JsonSchemaMixin
from parallax_ai.clients import ParallaxOpenAIClient

class Agent:
    def __init__(
        self, 
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        output_structure = None,
        max_tries: int = 5,
        **kwargs,
    ):
        self.model = model
        self.max_tries = max_tries
        self.system_prompt = system_prompt
        self.output_structure = output_structure
        self.client = ParallaxOpenAIClient(api_key=api_key, base_url=base_url)

    def get_output_schema(self):
        output_schema = self.output_structure.json_schema() if self.output_structure is not None else None
        output_schema.pop("$schema", None)
        output_schema.pop("description", None)
        return output_schema

    def __get_system_prompt(self):
        system_prompt = self.system_prompt
        if self.output_structure is not None:
            system_prompt = system_prompt + "\n\n" if system_prompt is not None else ""
            system_prompt += (
                "The output must be JSON that matches the following schema:\n"
                "{output_structure}"
            ).format(output_structure=json.dumps(self.get_output_schema()))
        return system_prompt

    def __convert_to_conversational_inputs(self, inputs):
        processed_inputs = []
        for input in inputs:
            if input is None:
                processed_inputs.append(None)
            elif isinstance(input, str):
                processed_inputs.append([{"role": "user", "content": input}])
            elif isinstance(input, list) and isinstance(input[0], dict):
                processed_inputs.append(input)
            else:
                raise ValueError(f"Unknown input type:\n{input}")
        return processed_inputs

    def __add_system_prompt_to_inputs(self, inputs):
        system_prompt = self.__get_system_prompt()

        processed_inputs = []
        for input in inputs:
            input = deepcopy(input)
            if input is None or system_prompt is None:
                processed_inputs.append(input)
            else:
                assert isinstance(input, list) and isinstance(input[0], dict)
                if input[0]["role"] == "system":
                    print("System prompt already exists, use the existing one. Note that the output_structure will not be added to the system prompt.")
                else:
                    input.insert(0, {"role": "system", "content": system_prompt})
                processed_inputs.append(input)
        return processed_inputs

    def _inputs_processing(self, inputs):
        # Ensure that inputs is a list
        if inputs is None or isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], dict)):
            inputs = [inputs]
        # Convert all inputs to conversational format
        inputs = self.__convert_to_conversational_inputs(inputs)
        # Add system prompt (if any) to the inputs
        return self.__add_system_prompt_to_inputs(inputs)
    
    def __get_text_output(self, output) -> str:
        return output.choices[0].message.content
    
    def __parse_and_validate_output(self, output: str) -> str|JsonSchemaMixin:
        if output is None:
            return None
        if self.output_structure is None:
            return output
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
            # # Validate the JSON object
            # jsonschema.validate(instance=output, schema=self.output_structure.json_schema())
            return self.output_structure.from_dict(output)
        except Exception:
            return None

    def _output_processing(self, output) -> str:
        if output is None:
            return None
        # Convert output object to text
        output = self.__get_text_output(output)
        # Parser and validate JSON output (if any)
        return self.__parse_and_validate_output(output)

    def run(
        self, 
        inputs, 
        verbose: bool = False,
        **kwargs,
    ) -> List[str]|List[JsonSchemaMixin]:
        inputs = self._inputs_processing(inputs)

        finished_outputs = {}
        unfinished_inputs = inputs
        for _ in range(self.max_tries):
            unfinished_indices = []
            outputs = self.client.run(inputs=unfinished_inputs, model=self.model, verbose=verbose, **kwargs)
            for i, output in enumerate(outputs):
                if unfinished_inputs[i] is None:
                    finished_outputs[i] = None
                else:
                    output = self._output_processing(output)
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
        **kwargs,
    ) -> Iterator[str]|Iterator[JsonSchemaMixin]:
        inputs = self._inputs_processing(inputs)

        current_index = 0
        finished_outputs = {}
        unfinished_indices = None
        unfinished_inputs = inputs
        for _ in range(self.max_tries):
            true_index_mapping = deepcopy(unfinished_indices) if unfinished_indices else []
            unfinished_indices = []
            for i, output in enumerate(self.client.irun(inputs=unfinished_inputs, model=self.model, **kwargs)):
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
                    output = self._output_processing(output)
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
        **kwargs,
    ) -> Iterator[Tuple[int, str]]|Iterator[Tuple[int, JsonSchemaMixin]]:
        inputs = self._inputs_processing(inputs)

        unfinished_indices = None
        unfinished_inputs = inputs
        for _ in range(self.max_tries):
            true_index_mapping = deepcopy(unfinished_indices) if unfinished_indices else []
            unfinished_indices = []
            for i, output in self.client.irun_unordered(inputs=unfinished_inputs, model=self.model, **kwargs):
                if unfinished_inputs[i] is None:
                    yield (i, None)
                else:
                    # Convert to true index
                    if len(true_index_mapping) > 0:
                        i = true_index_mapping[i]
                    # Process output
                    output = self._output_processing(output)
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


if __name__ == "__main__":
    from time import time
    from random import randint
    from typing import Literal
    from dataclasses import dataclass
    from dataclasses_jsonschema import JsonSchemaMixin

    @dataclass
    class OutputStructure(JsonSchemaMixin):
        name: str
        age: int
        gender: Literal["Male", "Female"]

    agent = Agent(
        model="google/gemma-3-27b-it",
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        output_structure=OutputStructure,
        max_tries=5,
    )
    print(agent.get_output_schema())

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
    print(f"Error: {error_count}")
    print(f"First Output Time: {first_output_time:4f}")
    print(f"Max Iteration Time: {max_iteration_time:4f}")
    print()