import json
from copy import deepcopy
from typing import List, Tuple, Optional, Iterator
from dataclasses_jsonschema import JsonSchemaMixin
from parallax_ai.clients import ParallaxOpenAIClient
from parallax_ai.agents.model_context import ModelContext, Field


class Agent:
    def __init__(
        self, 
        model: str,
        input_structure = None,     # If input_structure is provided, inputs must be instances of input_structure
        output_structure = None,    # If output_structure is provided, outputs will be validated against output_structure and converted to instances of output_structure
        model_context: Optional[ModelContext] = None,   # This allows dynamic and trainable contexts to be used as system prompts
        system_prompt: Optional[str] = None,   # Deprecated, use model_context instead. If both are provided, model_context will be used.
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tries: int = 5,
        **kwargs,
    ):
        if system_prompt is not None and model_context is not None and model_context.system_prompt is not None:
            print("Warning: Both system_prompt and model_context.system_prompt are provided. system_prompt will be ignored.")
        if model_context is None:
            model_context = ModelContext()
        if system_prompt is not None and model_context.system_prompt is None:
            # Create a ModelContext with the provided system_prompt
            model_context.system_prompt = [Field(name="system_prompt", content=system_prompt)]

        self.model = model
        self.max_tries = max_tries
        self.model_context = model_context
        self.input_structure = input_structure
        self.output_structure = output_structure
        self.client = ParallaxOpenAIClient(api_key=api_key, base_url=base_url)

    def __convert_to_conversational_inputs(self, inputs):
        processed_inputs = []
        for input in inputs:
            if input is None:
                processed_inputs.append(None)
            elif isinstance(input, str):
                processed_inputs.append([{"role": "user", "content": input}])
            elif isinstance(input, list) and isinstance(input[0], dict):
                processed_inputs.append(input)
            elif self.input_structure is not None and isinstance(input, self.input_structure):
                processed_inputs.append([{"role": "user", "content": self.model_context.render_input(input)}])
            else:
                raise ValueError(f"Unknown input type:\n{input}")
        return processed_inputs

    def __add_system_prompt_to_inputs(self, inputs, model_context: ModelContext = None):
        if model_context is None:
            model_context = self.model_context
        system_prompt = model_context.render_system_prompt(self.output_structure)

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

    def _inputs_processing(self, inputs, model_context: ModelContext = None):
        # Ensure that inputs is a list
        if inputs is None or isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], dict)) or (self.input_structure is not None and isinstance(inputs, self.input_structure)):
            inputs = [inputs]
        # Convert all inputs to conversational format
        inputs = self.__convert_to_conversational_inputs(inputs)
        # Add system prompt (if any) to the inputs
        return self.__add_system_prompt_to_inputs(inputs, model_context=model_context)
    
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
            # Validate the JSON object and convert to output_structure instance
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

    def parallel_run(
        self, 
        inputs, 
        model_contexts: List[ModelContext],
        verbose: bool = False,
        **kwargs,
    ) -> List[List[str]|List[JsonSchemaMixin]]:
        n = len(inputs)

        processed_inputs = []
        for model_context in model_contexts:
            processed_inputs.extend(self._inputs_processing(inputs, model_context=model_context))

        finished_outputs = {}
        unfinished_inputs = processed_inputs
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
            unfinished_inputs = [processed_inputs[i] for i in unfinished_indices]
        finished_outputs = [finished_outputs[i] if i in finished_outputs else None for i in range(len(processed_inputs))]
        outputs = [finished_outputs[i:i+n] for i in range(0, len(processed_inputs), n)]
        return outputs

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

    @dataclass
    class InputStructure(JsonSchemaMixin):
        topic: str
    print(InputStructure.json_schema())
    
    @dataclass
    class OutputStructure(JsonSchemaMixin):
        name: str
        age: int
        gender: Literal["Male", "Female"]
    print(OutputStructure.json_schema())

    agent = Agent(
        model="google/gemma-3-27b-it",
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        input_structure=InputStructure,
        output_structure=OutputStructure,
        model_context=ModelContext(
            system_prompt=[
                Field(name="task_definition", content="Given a topic, generate a list of people related to the topic.", title="Task Definition", trainable=False),
                Field(name="method", content="Think step by step before answering.", title="Methodology", trainable=True),
            ],
        ),
        max_tries=5,
    )
    print(agent.get_output_schema())

    inputs = [f"Generate a list of {randint(3, 20)} Thai singers" for _ in range(1000)]
    
    start_time = time()
    error_count = 0
    for i, output in enumerate(agent.run(inputs)):
        print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}s\nInput: {inputs[i]}\nOutput: {output}")
        if output is None:
            error_count += 1
    print(f"Error: {error_count}")
    print()
    
    prev_time = None
    start_time = time()
    error_count = 0
    max_iteration_time = 0
    for i, output in enumerate(agent.irun(inputs)):
        iteration_time = 0
        if prev_time:
            iteration_time = time() - prev_time
            if iteration_time > max_iteration_time:
                max_iteration_time = iteration_time
        if i == 0:
            first_output_time = time() - start_time
        print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}s ({iteration_time:.4f}s)\nInput: {inputs[i]}\nOutput: {output}")
        if output is None:
            error_count += 1
        prev_time = time()
    print(f"Error: {error_count}")
    print(f"First Output Time: {first_output_time:4f}")
    print(f"Max Iteration Time: {max_iteration_time:4f}")
    print()

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