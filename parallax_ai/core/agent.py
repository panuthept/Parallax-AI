import json
import random
from copy import deepcopy
from .client import ParallaxClient
from collections import defaultdict
from .engine import ParallaxEngine, Job
from ..utilities import type_validation, generate_session_id
from typing import Optional, Union, Literal, Tuple, List, Dict, Any, get_origin, get_args


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

    def fetch_or_init_conversations(self, inputs, system_prompt: Optional[str] = None) -> Tuple[List[str], List[Any], List[List[dict]]]:
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        session_ids = []
        new_inputs = []
        conversations = []
        for input in inputs:
            # Get session_id
            if isinstance(input, tuple):
                session_id, input = input
            else:
                session_id = generate_session_id()
            new_inputs.append(input)
            session_ids.append(session_id)
            # Init and fetch conversations
            if session_id not in self.sessions:
                self.sessions[session_id] = [] if system_prompt is None else [{"role": "system", "content": system_prompt}]
            conversation = deepcopy(self.sessions[session_id])
            conversations.append(conversation)
            # Update session_running_number
            if session_id not in self.session_running_number:
                self.session_running_number[session_id] = 0
            self.session_running_number[session_id] = self.running_number
            self.running_number += 1
        # Clear sessions
        self.ensure_max_sessions()
        return session_ids, new_inputs, conversations
    
    def create_branch(self, session_id: str):
        for i in range(self.max_branch):
            new_session_id = f"{session_id}/{i}"
            if new_session_id not in self.sessions:
                self.sessions[new_session_id] = deepcopy(self.sessions[session_id])
                return new_session_id
        raise ValueError("Fail to create new branch session due to exceeding max_branch limit.")
    
    def update_user(self, session_id, input):
        if self.max_sessions == 0:
            return session_id
        
        assert isinstance(input, list) and len(input) > 0 and input[-1]["role"] == "user", "Input must be a list of messages with the last message from user."
        assert session_id in self.sessions, f"Not found session id: {session_id}. Please ensure that min_sessions is not too small (must be larger than batch size)."
        if len(self.sessions[session_id]) > 0:
            if self.sessions[session_id][-1]["role"] == "user":
                # Create branch session_id
                session_id = self.create_branch(session_id)
        self.sessions[session_id].append({"role": "user", "content": input[-1]["content"]})
        return session_id
    
    def update_assistant(self, session_id, output):
        if self.max_sessions == 0:
            return session_id
        
        assert session_id in self.sessions, f"Not found session id: {session_id}. Please ensure that min_sessions is not too small (must be larger than batch size)."
        if self.sessions[session_id][-1]["role"] == "assistant":
            # Create branch session_id
            session_id = self.create_branch(session_id)
        self.sessions[session_id].append({"role": "assistant", "content": output.choices[0].message.content})
        return session_id


class InputProcessor:
    def __init__(
        self,
        input_structure: Optional[Union[Dict, type]] = None,
        input_template: Optional[str] = None,
    ):
        self.input_structure = input_structure
        self.input_template = input_template

    def __render_input(self, input):
        # Extract only the keys that are in the input_structure
        assert self.input_structure is not None
        input = {key: value for key, value in input.items() if key in self.input_structure}
        # If not all keys are in the input, raise error
        for key in self.input_structure:
            assert key in input, f"key '{key}' missing from the inputs of the agent"
        # Check type of each value
        for key, value in input.items():
            if not type_validation(value, self.input_structure[key]):
                raise ValueError(f"Type of key '{key}' is not valid: expecting {self.input_structure[key]} but got {type(value)} from the agent named: {self.name}")

        if self.input_template is None:
            return "\n\n".join([f'{key.replace("_", " ").capitalize()}:\n{value}' for key, value in input.items()])
        else:
            return self.input_template.format(**input)
        
    def __ensure_inputs_format(self, inputs):
        if self.input_structure is not None: 
            # input_structure can be either dict or type
            if isinstance(inputs, list):
                for inp in inputs:
                    if inp is None:
                        continue
                    if isinstance(self.input_structure, dict):
                        assert isinstance(inp, dict), "Invalid inputs"
                    else:
                        assert type_validation(inp, self.input_structure), "Invalid inputs"
            else:
                if isinstance(self.input_structure, dict):
                    assert isinstance(inputs, dict), "Invalid inputs"
                    inputs = [inputs]
                else:
                    assert type_validation(inputs, self.input_structure), "Invalid inputs"
                    inputs = [inputs]
        else:
            if isinstance(inputs, list):
                # Must be [None], [str]
                for inp in inputs:
                    # Must be None, str
                    assert isinstance(inp, str) or inp is None, "Invalid inputs"
            else:
                # Must be None, str
                assert isinstance(inputs, str) or inputs is None, "Invalid inputs"
                inputs = [inputs]
        return inputs

    def __convert_to_conversational_inputs(self, inputs, conversations):
        new_conversations = []
        for input, prev_conversation in zip(inputs, conversations):
            prev_conversation = deepcopy(prev_conversation)
            if input is None:
                new_conversations.append(input)
                continue
            # Convert input to conversational
            if isinstance(self.input_structure, dict):
                input = {"role": "user", "content": self.__render_input(input)}
            else:
                assert isinstance(input, str), f"Input must be string. Got {input}"
                input = {"role": "user", "content": input}
            if len(prev_conversation) > 0:
                assert prev_conversation[-1]["role"] != "user", "The last message in the conversation must not be from user."
            prev_conversation.append(input)
            new_conversations.append(prev_conversation)
        return new_conversations

    def __call__(self, inputs, conversations):
        # Ensure inputs type and convert inputs to list of needed
        inputs = self.__ensure_inputs_format(inputs)
        # Convert all inputs to conversational format
        return self.__convert_to_conversational_inputs(inputs, conversations)
    

class OutputProcessor:
    def __init__(self, output_structure: Optional[Union[Dict, type]] = None):
        self.output_structure = output_structure

    def __get_text_output(self, output) -> str:
        return output.choices[0].message.content
    
    def __parse_and_validate_output(self, output: str) -> Union[Dict, str]:
        if self.output_structure is None:
            return output
        
        try:
            if (isinstance(self.output_structure, list) and isinstance(self.output_structure[0], dict)) or isinstance(self.output_structure, dict):
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
                if isinstance(self.output_structure, list):
                    assert isinstance(output, list)
                    outputs = output
                    valid_outputs = []
                    # Check if all keys are in the output
                    for key in self.output_structure[0]:
                        for output in outputs:
                            if key in output:
                                # Check if all values are valid
                                type_validation(output[key], self.output_structure[0][key], raise_error=True)
                                valid_outputs.append(output)
                    if len(valid_outputs) == 0:
                        raise ValueError("No valid output found")
                    output = valid_outputs
                else:
                    assert isinstance(output, dict)
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
        output = self.__parse_and_validate_output(output)
        return output
        

class Agent:
    def __init__(
        self, 
        model: str,
        input_structure: Optional[Union[Dict, type]] = None,
        output_structure: Optional[Union[List[Dict], Dict, type]] = None,
        input_template: Optional[str] = None,
        system_prompt: Optional[str] = None,
        conversational_agent: bool = False,
        min_sessions: int = 100000,
        max_sessions: int = 1000000,
        max_tries: int = 10,
        dismiss_none_output: bool = False,
        client: Optional[ParallaxClient] = None,
        **kwargs,
    ):  
        if input_structure is not None:
            assert isinstance(input_structure, dict) or ((hasattr(input_structure, '__origin__') and get_origin(input_structure) == Literal)), f"input_structure only support dictionary or Literal type. Got {input_structure}."
        if output_structure is not None:
            assert (isinstance(output_structure, list) and isinstance(output_structure[0], dict)) or isinstance(output_structure, dict) or ((hasattr(output_structure, '__origin__') and get_origin(output_structure) == Literal)), f"output_structure only support list of dictionary,  dictionary or Literal type. Got {output_structure}."

        self.model = model
        self.input_structure = input_structure
        self.output_structure = output_structure
        self.input_template = input_template
        self.system_prompt = system_prompt
        self.conversational_agent = conversational_agent

        self.conversation_memory = ConversationMemory(
            min_sessions=min_sessions if conversational_agent else 0,
            max_sessions=max_sessions if conversational_agent else 0,
        )
        self.input_processor = InputProcessor(
            input_structure=input_structure,
            input_template=input_template,
        )
        self.output_processor = OutputProcessor(
            output_structure=output_structure,
        )
        self.client = ParallaxClient(**kwargs) if client is None else client
        self.engine = ParallaxEngine(
            client=self.client, 
            max_tries=max_tries,
            dismiss_none_output=dismiss_none_output,
        )

    def get_system_prompt(self):
        system_prompt = self.system_prompt
        if system_prompt is None:
            return None

        output_structure = self.output_structure
        if (isinstance(output_structure, list) and isinstance(output_structure[0], dict)) or isinstance(output_structure, dict):
            is_list = isinstance(output_structure, list)
            if is_list:
                output_structure = output_structure[0]

            schema = json.dumps({k: str(v).replace("typing.", "").replace("<class '", "").replace("'>", "") for k, v in output_structure.items()})
            items = []
            for item in schema[1:-1].split("\", \""):
                if not item.startswith('"'):
                    item = '"' + item
                if not item.endswith('"'):
                    item = item + '"'
                key, value = item.split(": ")
                value = value[1:-1]
                item = f"{key}: {value}"
                items.append(item)
            schema = "{" + ", ".join(items) + "}"
            system_prompt = system_prompt + "\n\n" if system_prompt is not None else ""
            if is_list:
                system_prompt += (
                    "The final output must be a JSON array of objects that exactly match the following schema:\n"
                    "```json\n"
                    "[{output_structure}]\n"
                    "```"
                ).format(output_structure=schema)
            else:
                system_prompt += (
                    "The final output must be a single JSON that exactly matches the following schema:\n"
                    "```json\n"
                    "{output_structure}\n"
                    "```"
                ).format(output_structure=schema)
        elif get_origin(output_structure) == Literal:
            keywords = "\n".join(list(get_args(output_structure)))
            system_prompt = system_prompt + "\n\n" if system_prompt is not None else ""
            system_prompt += (
                "The final output must be a one of the following keyword:\n"
                "{keywords}"
            ).format(keywords=keywords)
        return system_prompt
    
    def _create_jobs(
        self, 
        inputs: List[Tuple[str, Any]],
        progress_name: Optional[str] = None
    ) -> List[Job]:
        # Fetch previous conversations or init new conversations (system prompt will be added here)
        session_ids, inputs, conversations = self.conversation_memory.fetch_or_init_conversations(
            inputs, system_prompt=self.get_system_prompt()
        )
        if len(inputs) == 0:
            return []
        # Process inputs (Ensure input format and convert to conversational format)
        inputs = self.input_processor(inputs, conversations)
        # Update conversation memory with user inputs
        for session_id, inp in zip(session_ids, inputs):
            if inp is not None:
                self.conversation_memory.update_user(session_id, inp)
        return [
            Job(
                inp=inp, 
                model=self.model, 
                session_id=session_id,
                output_processor=self.output_processor,
                progress_name=progress_name,
            ) for session_id, inp in zip(session_ids, inputs)
        ]

    def _run(
        self, 
        inputs: List[Tuple[str, Any]],
        progress_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[str], List[Any]]:
        # Create jobs
        jobs = self._create_jobs(inputs, progress_name)
        if len(jobs) == 0:
            return [], []
        # Process the jobs by the ParallaxEngine with retries
        jobs = self.engine(jobs, **kwargs)
        # Get outputs in the original order
        outputs = [job.output for job in jobs]
        # Update conversation memory with assistant outputs
        for job in jobs:
            if job.output is not None:
                job.session_id = self.conversation_memory.update_assistant(job.session_id, job.output)
        session_ids = [job.session_id for job in jobs]
        return session_ids, outputs
    
    def input_transformation(self, inputs, progress_name: Optional[str] = None):
        return inputs, progress_name
    
    def output_transformation(self, outputs):
        return outputs
    
    def run(
        self, 
        inputs: List[Union[Tuple[str, Any], Tuple[str, Any]]],
        progress_name: Optional[str] = None,
        **kwargs,
    ) -> List[Tuple[str, Any]]:
        inputs, progress_name = self.input_transformation(inputs, progress_name)

        session_ids, outputs = self._run(inputs, progress_name=progress_name, **kwargs)
        if self.conversational_agent:
            outputs = [(session_id, output) for session_id, output in zip(session_ids, outputs)]
        else:
            outputs = [output for output in outputs]

        return self.output_transformation(outputs)


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
