import json
import random
from openai import OpenAI
from ..dataclasses import Job
from dataclasses import dataclass
from .base_module import BaseModule
from ..utilities import type_validation, get_dummy_output
from typing import Any, Literal, List, Optional, get_origin, get_args


def prompt_completions(inputs: dict):
    # Sample a model address
    model_address = random.choice(inputs["model_addresses"])
    api_key = model_address.get("api_key")
    base_url = model_address.get("base_url")
    # Get raw response from the model
    client = OpenAI(api_key=api_key, base_url=base_url)
    output = client.completions.create(
        model=inputs["model"],
        prompt=inputs["prompt"],
        **inputs["kwargs"]
    )
    return output

def chat_completions(inputs: dict):
    # Sample a model address
    model_address = random.choice(inputs["model_addresses"])
    api_key = model_address.get("api_key")
    base_url = model_address.get("base_url")
    # Get raw response from the model
    client = OpenAI(api_key=api_key, base_url=base_url)
    output = client.chat.completions.create(
        model=inputs["model"],
        messages=inputs["messages"],
        **inputs["kwargs"]
    )
    return output

def output_verify_and_parsing(output, output_structure: Any) -> Any:
    # Extract the content
    output = output.choices[0].message.content
    if output_structure is not None:
        # Parse the output
        if (isinstance(output_structure, list) and isinstance(output_structure[0], dict)) or isinstance(output_structure, dict):
            if "```json" in output:
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
            if isinstance(output_structure, list):
                assert isinstance(output, list)
                outputs = output
                valid_outputs = []
                # Check if all keys are in the output
                for output in outputs:
                    is_valid = True
                    for key in output_structure[0]:
                        if key not in output:
                            is_valid = False
                            break
                        # Check if all values are valid
                        if not type_validation(output[key], output_structure[0][key]):
                            is_valid = False
                            break
                    if is_valid:
                        valid_outputs.append(output)
                if len(valid_outputs) == 0:
                    raise ValueError("No valid output found")
                output = valid_outputs
            else:
                assert isinstance(output, dict)
                # Check if all keys are in the output
                for key in output_structure:
                    if key not in output:
                        raise ValueError(f"Key {key} is missing in the output")
                # Check if all values are valid
                for key, value in output.items():
                    type_validation(value, output_structure[key], raise_error=True)
        elif get_origin(output_structure) == Literal:
            keywords = list(get_args(output_structure))
            if output not in keywords:
                raise ValueError(f"Output '{output}' is not in the allowed keywords: {keywords}")
        else:
            type_validation(output, output_structure, raise_error=True)
    return output

def agent_completions(inputs: dict):
    raw_output = chat_completions(inputs)
    parsed_output = output_verify_and_parsing(raw_output, inputs.get("output_structure"))
    return parsed_output

@dataclass
class ModelSpec:
    model_name: str
    prompt_template: Optional[str] = None

@dataclass
class AgentSpec(ModelSpec):
    model_name: str
    system_prompt: Optional[str] = None
    input_structure: Optional[dict] = None
    output_structure: Optional[dict|List[dict]] = None

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

@dataclass
class AgentModule(BaseModule):
    spec: AgentSpec = None
    max_retries: int = 10
    kwargs: Optional[dict] = None

    def get_agent_input(self, module_input: dict) -> dict:
        # Ensure input matches the defined structure
        if self.spec.input_structure is not None:
            # Filter irrelevant fields
            module_input = {k: v for k, v in module_input.items() if k in self.spec.input_structure}
            for key in self.spec.input_structure.keys():
                assert type_validation(module_input[key], self.spec.input_structure[key]), "Invalid inputs. Expected structure: {self.spec.input_structure}, but got: {module_input}"
        # Convert input to conversational format
        messages = []
        # Add system prompt if available
        system_prompt = self.spec.get_system_prompt()
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        # Add input prompt as user message
        if self.spec.prompt_template is not None:
            content = self.spec.prompt_template.format(**module_input)
        else:
            content = "\n\n".join([f'{key.replace("_", " ").capitalize()}:\n{value}' for key, value in module_input.items()])
        messages.append({"role": "user", "content": content})
        return messages
    
    def get_executor_input(self, module_input: dict) -> dict:
        assert self.worker_nodes is not None, "worker_nodes must be provided for AgentModule."
        assert self.spec.model_name in self.worker_nodes, f"Model addresses for model '{self.spec.model_name}' not found in worker_nodes."

        agent_input = self.get_agent_input(module_input)
        executor_input = {
            "messages": agent_input,
            "model": self.spec.model_name,
            "model_addresses": self.worker_nodes[self.spec.model_name],
            "output_structure": self.spec.output_structure,
            "max_retries": self.max_retries,
            "kwargs": self.kwargs or {}
        }
        return executor_input

    def _create_job(self, instance_id: str, module_input: dict) -> Job:
        return Job(
            module_input=module_input,
            executor_func=agent_completions,
            executor_input=self.get_executor_input(module_input),
            executor_default_output=get_dummy_output(self.spec.output_structure),
            instance_id=instance_id,
            module_name=self.name,
            progress_name=self.progress_name
        )