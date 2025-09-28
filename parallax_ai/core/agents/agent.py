import json
from copy import deepcopy
from .model_context import ModelContext
from parallax_ai.core import ParallaxClient
from parallax_ai.utilities import type_validation, generate_session_id
from typing import Literal, Tuple, List, Dict, Optional, Any, get_origin


class ConversationMemory:
    def __init__(
        self, 
        min_sessions: int = 100000,
        max_sessions: int = 1000000,
        max_branch: int = 1000000,
    ):
        self.min_sessions = min_sessions
        self.max_sessions = max_sessions
        self.max_branch = max_branch

        self.sessions: Dict[str, List] = {}
        self.running_number = 0
        self.session_running_number = {}

    def ensure_max_sessions(self):
        if len(self.sessions) > self.max_sessions:
            # Sort session_ids by their running_number
            sorted_sessions = sorted(self.session_running_number.items(), key=lambda x:x[1], reverse=True)
            # Remove exceed sessions
            for session_id, _ in sorted_sessions[self.min_sessions:]:
                del self.sessions[session_id]
                del self.session_running_number[session_id]

    def fetch_or_init_conversations(self, inputs, system_prompt: Optional[str] = None) -> Tuple[List[str], List[List[dict]]]:
        session_ids = []
        conversations = []
        for input in inputs:
            # Get session_id
            if isinstance(input, tuple):
                session_id, input = input
            else:
                session_id = generate_session_id()
            session_ids.append(session_id)
            # Init and fetch conversations
            if session_id not in self.sessions:
                self.sessions[session_id] = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
            conversation = self.sessions[session_id]
            conversations.append(conversation)
            # Update session_running_number
            if session_id not in self.session_running_number:
                self.session_running_number[session_id] = 0
            self.session_running_number = self.running_number
            self.running_number += 1
        # Clear sessions
        self.ensure_max_sessions()
        return session_ids, conversations
    
    def create_branch(self, session_id: str):
        for i in range(self.max_branch):
            new_session_id = f"{session_id}/{i}"
            if new_session_id not in self.sessions:
                self.sessions[new_session_id] = deepcopy(self.sessions[session_id])
                return new_session_id
        raise ValueError("Fail to create new branch session due to exceeding max_branch limit.")
    
    def update(self, session_id, output):
        assert session_id in self.sessions, f"Not found session id: {session_id}. Please ensure that min_sessions is not too small (must be larger than batch size)."
        if self.sessions[session_id][-1]["role"] == "assistant":
            # Create branch session_id
            session_id = self.create_branch(session_id)
        self.sessions[session_id].append({"role": "assistant", "content": output.choices[0].message.content})
            


class InputProcessor:
    def __init__(
        self,
        input_structure: Optional[dict|type] = None,
        output_structure: Optional[dict|type] = None,
        input_template: Optional[str] = None,
    ):
        self.input_structure = input_structure
        self.output_structure = output_structure
        self.input_template = input_template

    def __render_input(self, input):
        # Extract only the keys that are in the input_structure
        assert self.input_structure is not None
        input = {key: value for key, value in input.items() if key in self.input_structure}
        # If not all keys are in the input, raise error
        for key in self.input_structure:
            assert key in input, f"key '{key}' missing from the inputs of the agent named: {self.name}"
        # Check type of each value
        for key, value in input.items():
            if not type_validation(value, self.input_structure[key]):
                raise ValueError(f"Type of key '{key}' is not valid: expecting {self.input_structure[key]} but got {type(value)} from the agent named: {self.name}")

        if self.input_template is None:
            return "\n\n".join([f"{key.replace("_", " ").capitalize()}:\n{value}" for key, value in input.items()])
        else:
            return self.input_template.format(**input)
        
    def __ensure_inputs_format(self, inputs):
        if self.input_structure is not None: 
            # input_structure can be either dict or type
            if isinstance(inputs, list):
                if isinstance(self.input_structure, dict):
                    assert isinstance(inputs[0], dict), "Invalid inputs"
                else:
                    assert type_validation(inputs[0], self.input_structure), "Invalid inputs"
            else:
                if isinstance(self.input_structure, dict):
                    assert isinstance(inputs, dict), "Invalid inputs"
                    inputs = [inputs]
                else:
                    assert type_validation(inputs, self.input_structure), "Invalid inputs"
                    inputs = [inputs]
        else:
            if isinstance(inputs, list):
                # Can be [None], [str], [{'role': str, 'content': str}], [[{'role': str, 'content': str}]]
                if isinstance(inputs[0], list):
                    # Must be [[{'role': str, 'content': str}]]
                    assert isinstance(inputs[0][0], dict) and "role" in inputs[0][0] and "content" in inputs[0][0], "Invalid inputs"
                else:
                    # Can be [None], [str], [{'role': str, 'content': str}]
                    if isinstance(inputs[0], dict):
                        # Must be [{'role': str, 'content': str}]
                        assert "role" in inputs[0] and "content" in inputs[0], "Invalid inputs"
                        inputs = [inputs]
                    else:
                        # Must be [None], [str]
                        assert isinstance(inputs, str) or inputs is None, "Invalid inputs"
            else:
                # Must be None, str
                assert isinstance(inputs, str) or inputs is None, "Invalid inputs"
                inputs = [inputs]
        return inputs

    def __convert_to_conversational_inputs(self, inputs, conversations):
        new_conversations = []
        for input, prev_conversation in zip(inputs, conversations):
            if input is None:
                new_conversations.append(input)
                continue
            # Convert input to conversational
            if isinstance(self.input_structure, dict):
                input = {"role": "user", "content": self.__render_input(input)}
            else:
                assert isinstance(input, str), "Input must be string."
                input = {"role": "user", "content": input}
            prev_conversation.append(input)
            new_conversations.append(prev_conversation)
        return new_conversations

    # def __add_system_prompt_to_inputs(self, inputs):
    #     system_prompt = self.model_context.render_system_prompt(self.output_structure)

    #     processed_inputs = []
    #     for input in inputs:
    #         input = deepcopy(input)
    #         if input is None or system_prompt is None:
    #             processed_inputs.append(input)
    #         else:
    #             if input[0]["role"] != "system":
    #                 print("System prompt already exists, use the existing one. Note that the output_structure will not be added to the system prompt.")
    #             else:
    #                 input.insert(0, {"role": "system", "content": system_prompt})
    #             processed_inputs.append(input)
    #     return processed_inputs

    def __call__(self, inputs, conversations):
        # Ensure inputs type and convert inputs to list of needed
        inputs = self.__ensure_inputs_format(inputs)
        # Convert all inputs to conversational format
        return self.__convert_to_conversational_inputs(inputs, conversations)
        # # Add system prompt (if any) to the inputs
        # return self.__add_system_prompt_to_inputs(inputs)
    

class OutputProcessor:
    def __init__(self, output_structure: Optional[dict|type] = None):
        self.output_structure = output_structure

    def __get_text_output(self, output) -> str:
        return output.choices[0].message.content
    
    def __parse_and_validate_output(self, output: str) -> str|dict:
        if output is None:
            return None
        
        if self.output_structure is None:
            return output
        
        try:
            if isinstance(self.output_structure, dict):
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
                # Check if all keys are in the output
                for key in self.output_structure:
                    if key not in output:
                        raise ValueError(f"Key {key} is missing in the output")
                # Check if all values are valid
                for key, value in output.items():
                    type_validation(value, self.output_structure[key], raise_error=True)
            else:
                type_validation(output, self.output_structure, raise_error=True)
            return output
        except Exception:
            return None

    def __call__(self, output) -> str:
        if output is None:
            return None
        # Convert output object to text
        output = self.__get_text_output(output)
        # Parser and validate JSON output (if any)
        return self.__parse_and_validate_output(output)
        

class Agent:
    def __init__(
        self, 
        model: str,
        name: Optional[str] = None,
        input_structure: Optional[dict|type] = None,
        output_structure: Optional[dict|type] = None,
        input_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        min_sessions: int = 100000,
        max_sessions: int = 1000000,
        max_tries: int = 5,
        **kwargs,
    ):  
        if input_structure is not None:
            assert isinstance(input_structure, dict) or get_origin(input_structure) == Literal, "input_structure only support dictionary of Literal."
        if output_structure is not None:
            assert isinstance(output_structure, dict) or get_origin(input_structure) == Literal, "output_structure only support dictionary of Literal."

        self.model = model
        self.name = name
        self.max_tries = max_tries
        self.input_structure = input_structure
        self.output_structure = output_structure
        self.input_template = input_template
        self.system_prompt = system_prompt

        self.model_context = ModelContext(
            system_prompt=system_prompt
        )
        self.conversation_memory = ConversationMemory(
            min_sessions=min_sessions,
            max_sessions=max_sessions,
        )
        self.input_processor = InputProcessor(
            input_structure=input_structure,
            output_structure=output_structure,
            input_template=input_template,
            system_prompt=system_prompt,
        )
        self.output_processor = OutputProcessor(
            output_structure=output_structure,
        )
        self.client = ParallaxClient(**kwargs)

    def get_system_prompt(self):
        return self.model_context.render_system_prompt(self.output_structure)

    def _run(
        self, 
        inputs: List[Tuple[str, Any]],
        verbose: bool = False,
        desc: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[str], List[Any]]:
        session_ids, conversations = self.conversation_memory.fetch_or_init_conversations(
            inputs, system_prompt=self.get_system_prompt()
        )
        inputs = self.input_processor(inputs, conversations)

        finished_outputs = {}
        unfinished_inputs = inputs
        for _ in range(self.max_tries):
            unfinished_indices = []
            outputs = self.client.run(inputs=unfinished_inputs, model=self.model, verbose=verbose, desc=desc, **kwargs)
            for i, output in enumerate(outputs):
                if unfinished_inputs[i] is None:
                    finished_outputs[i] = None
                else:
                    processed_output = self.output_processor(output)
                    if processed_output is not None:
                        finished_outputs[i] = processed_output
                        session_ids[i] = self.conversation_memory.update(session_ids[i], output)
                    else:
                        unfinished_indices.append(i)
            if len(unfinished_indices) == 0:
                break
            unfinished_inputs = [inputs[i] for i in unfinished_indices]

        outputs = [finished_outputs[i] if i in finished_outputs else None for i in range(len(inputs))]
        return session_ids, outputs
    
    def run(
        self, 
        inputs: List[Tuple[str, Any]]|Tuple[str, Any],
        verbose: bool = False,
        desc: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[str, Any]]:
        if not isinstance(inputs, list):
            inputs = [inputs]
        session_ids, outputs = self._run(inputs, verbose=verbose, desc=desc, *kwargs)
        return [(session_id, output) for session_id, output in zip(session_ids, outputs)]


if __name__ == "__main__":
    from time import time
    from random import randint
    from typing import Literal

    agent = Agent(
        model="google/gemma-3-27b-it",
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
        output_structure={"name": str, "age": int, "gender": Literal["Male", "Female"]},
        system_prompt=(
            "Given a topic, generate a list of people related to the topic.\n"
            "Think step by step before answering."
        ),
        max_tries=5,
    )

    # Multi-turn interaction
    inputs = [f"Generate a list of {randint(3, 20)} Thai singers" for _ in range(1000)]

    start_time = time()
    error_count = 0
    next_inputs = []
    for i, (session_id, output) in enumerate(agent.run(inputs)):
        print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}s\nInput: {inputs[i]}\nOutput: {output}")
        if output is None:
            error_count += 1
        next_inputs.append((session_id, output))
    print(f"Error: {error_count}")
    print()

    for i, (session_id, output) in enumerate(agent.run(next_inputs)):
        print(f"[{i + 1}] elapsed time: {time() - start_time:.4f}s\nInput: {inputs[i]}\nOutput: {output}")
        if output is None:
            error_count += 1
    print(f"Error: {error_count}")
    print()
