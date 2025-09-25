import json
from copy import deepcopy
from typing_validation import validate
from typing import List, Optional, Callable
from parallax_ai.core import ParallaxClient
from .model_context import ModelContext, Field
from dataclasses_jsonschema import JsonSchemaMixin


class Agent:
    def __init__(
        self, 
        model: str,
        name: Optional[str] = None,
        input_structure: Optional[dict|type] = None,
        output_structure: Optional[dict|type] = None,
        input_template: Optional[str] = None,
        input_transformation: Optional[Callable] = None,
        output_transformation: Optional[Callable] = None,
        model_context: Optional[ModelContext] = None,   # Deprecated
        system_prompt: Optional[str] = None,
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
        if input_structure is not None:
            assert isinstance(input_structure, dict) or isinstance(input_structure, type)
        if output_structure is not None:
            assert isinstance(output_structure, dict) or isinstance(output_structure, type)

        self.model = model
        self.name = name
        self.max_tries = max_tries
        self.model_context = model_context
        self.input_structure = input_structure
        self.output_structure = output_structure
        self.input_template = input_template
        self.input_transformation = input_transformation
        self.output_transformation = output_transformation
        self.client = ParallaxClient(**kwargs)

    def __render_input(self, input):
        # Extract only the keys that are in the input_structure
        assert self.input_structure is not None
        input = {key: value for key, value in input.items() if key in self.input_structure}
        # If not all keys are in the input, raise error
        for key in self.input_structure:
            assert key in input, f"key '{key}' missing from the inputs of the agent named: {self.name}"
        # Check type of each value
        for key, value in input.items():
            try:
                validate(value, self.input_structure[key])
            except TypeError:
                raise ValueError(f"Type of key '{key}' is not valid: expecting {self.input_structure[key]} but got {type(value)} from the agent named: {self.name}")

        if self.input_template is None:
            return "\n\n".join([f"{key.replace("_", " ").capitalize()}:\n{value}" for key, value in input.items()])
        else:
            return self.input_template.format(**input)

    def __convert_to_conversational_inputs(self, inputs):
        processed_inputs = []
        for input in inputs:
            if input is None:
                processed_inputs.append(None)
            else:
                if self.input_structure is None:
                    if isinstance(input, str):
                        processed_inputs.append([{"role": "user", "content": input}])
                    elif isinstance(input, list) and isinstance(input[0], dict):
                        if "role" in input[0] and "content" in input[0]:
                            processed_inputs.append(input)
                        else:
                            processed_inputs.append([{"role": "user", "content": json.dumps(input, indent=4, ensure_ascii=False)}])
                    else:
                        raise ValueError(f"Unknown input type:\n{input}")
                else:
                    if isinstance(self.input_structure, type) and isinstance(input, self.input_structure):
                        processed_inputs.append([{"role": "user", "content": self.model_context.render_input(input)}])
                    elif isinstance(self.input_structure, dict) and isinstance(input, dict):
                        processed_inputs.append([{"role": "user", "content": self.__render_input(input)}])
                    else:
                        raise ValueError(f"Unknown input structure type:\n{self.input_structure}")
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
                if "role" in input[0] and "content" in input[0]:
                    if input[0]["role"] == "system":
                        print("System prompt already exists, use the existing one. Note that the output_structure will not be added to the system prompt.")
                    else:
                        input.insert(0, {"role": "system", "content": system_prompt})
                else:
                    input.insert(0, {"role": "system", "content": system_prompt})
                processed_inputs.append(input)
        return processed_inputs

    def _inputs_processing(self, inputs, model_context: ModelContext = None):
        # Ensure that inputs is a list
        if self.input_structure is not None and isinstance(self.input_structure, dict):
            if isinstance(inputs, dict):
                inputs = [inputs]
        else:
            if inputs is None or isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], dict)):
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
            # Fix \n problem in JSON
            output = "".join([line.strip() for line in output.split("\n")])
            # Parse the JSON object
            output = json.loads(output)
            # Validate the JSON object and convert to output_structure 
            if isinstance(self.output_structure, type):
                return self.output_structure.from_dict(output)
            elif isinstance(self.output_structure, dict):
                # Check if all keys are in the output
                for key in self.output_structure:
                    if key not in output:
                        raise ValueError(f"Key {key} is missing in the output")
                # Check if all values are valid
                for key, value in output.items():
                    validate(value, self.output_structure[key])
                return output
                
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
    ) -> List[List[str]|List[dict]|List[JsonSchemaMixin]]:
        inputs = self.input_transformation(inputs) if self.input_transformation is not None else inputs
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
        outputs = self.output_transformation(outputs) if self.output_transformation is not None else outputs
        return outputs

    def run(
        self, 
        inputs, 
        verbose: bool = False,
        desc: Optional[str] = None,
        **kwargs,
    ) -> List[str]|List[dict]|List[JsonSchemaMixin]:
        inputs = self.input_transformation(inputs) if self.input_transformation is not None else inputs
        inputs = self._inputs_processing(inputs)

        finished_outputs = {}
        unfinished_inputs = inputs
        for _ in range(self.max_tries):
            unfinished_indices = []
            outputs = self.client.run(inputs=unfinished_inputs, model=self.model, verbose=verbose, desc=desc, **kwargs)
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

        outputs = [finished_outputs[i] if i in finished_outputs else None for i in range(len(inputs))]
        outputs = self.output_transformation(outputs) if self.output_transformation is not None else outputs
        return outputs


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