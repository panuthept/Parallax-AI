import os
import yaml
import json
from collections import defaultdict
from parallax_ai.core.agents.agent import Agent
from parallax_ai.utilities import generate_session_id
from typing import Literal, Union, Tuple, List, Dict, Iterable, Callable, Optional, Any

class MultiAgent:
    def __init__(
        self,
        agents: Dict[str, Agent],
        pipeline: List[Union[str, Iterable[str]]] = None,
        auxiliary_functions: Optional[Dict[str, Callable]] = None,
        output_to_register: Optional[Dict[str, Dict[str, str]]] = None,
        input_transformation: Optional[Dict[str, Callable]] = None,
        output_transformation: Optional[Dict[str, Callable]] = None,
        logging_path: str = "./multiagent/logs.json",
    ):
        self.datapool = {}
        self.agents = agents
        self.pipeline = pipeline
        self.auxiliary_functions = auxiliary_functions if auxiliary_functions is not None else {}
        self.output_to_register = output_to_register if output_to_register is not None else {}
        self.input_transformation = input_transformation if input_transformation is not None else {}
        self.output_transformation = output_transformation if output_transformation is not None else {}
        self.logging_path = logging_path

    @classmethod
    def from_yaml(
        cls, 
        path: str,
        logging_path: str = "./multiagent/logs.json",
        **kwargs,
    ) -> "MultiAgent":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        agents = {}
        for name, config in data["agents"].items():
            # Replace config with kwargs (if any)
            config.update(kwargs)
            # Replace string with actual types or functions
            if "input_structure" in config:
                config["input_structure"] = {k: eval(v) for k, v in config["input_structure"].items()}
            if "output_structure" in config:
                config["output_structure"] = {k: eval(v) for k, v in config["output_structure"].items()}
            agents[name] = Agent(**config)
        return cls(agents=agents, logging_path=logging_path)

    def register_data(self, name: str, data: Any, allow_overwrite: bool = False):
        if name in self.datapool and not allow_overwrite:
            raise ValueError(f"Data with name '{name}' already exists. Set allow_overwrite=True to overwrite it.")
        self.datapool.update({name: data})

    def register_or_append_data(self, name: str, data: Any):
        if name in self.datapool:
            assert isinstance(self.datapool[name], list), f"Data with name '{name}' is not a list."
            if isinstance(data, list):
                self.datapool[name].extend(data)
            else:
                self.datapool[name].append(data)
        else:
            self.register_data(name, data)

    def logging_datapool(self):
        os.makedirs(os.path.dirname(self.logging_path), exist_ok=True)
        with open(self.logging_path, "w") as f:
            json.dump(self.datapool, f)

    def _run(
        self, 
        component_name: str, 
        inputs: List[Tuple[str, Any]], 
        logging: bool = False, 
        verbose: bool = False,
        debug: bool = False,
    ) -> List[Tuple[str, Any]]:
        if debug: print(f"Running agent/function: {component_name} with inputs:\n{inputs}")
        # Input transformation
        if component_name in self.input_transformation:
            inputs = self.input_transformation[component_name](inputs, self.datapool)
            if debug: print(f"Transformed inputs:\n{inputs}")
            
        # Run Agent or Function
        if component_name in self.agents:
            outputs: List[Tuple[str, Any]] = self.agents[component_name].run(
                inputs, debug=debug, verbose=verbose, desc="Running agent - " + component_name, max_tokens=2048,
            )
        else:
            outputs = self.auxiliary_functions(inputs)
        if debug: print(f"Raw outputs:\n{outputs}")

        # Filter out None outputs
        if self.agents[component_name].conversational_agent:
            valid_indices = [i for i, (session_id, output) in enumerate(outputs) if output is not None]
        else:
            valid_indices = [i for i, output in enumerate(outputs) if output is not None]
        inputs = [inputs[i] for i in valid_indices]
        outputs = [outputs[i] for i in valid_indices]

        # Output transformation
        if component_name in self.output_transformation:
            outputs = self.output_transformation[component_name](inputs, outputs, self.datapool)
            if debug: print(f"Transformed outputs:\n{outputs}")
        
        # Register outputs to datapool (If any)
        if component_name in self.output_to_register:
            for key_name, save_name in self.output_to_register[component_name].items():
                data = []
                for session_id, output in outputs:
                    if output is None:
                        continue
                    assert key_name in output, f"Key '{key_name}' not found in the outputs of the agent named: {component_name}"
                    data.append(output[key_name])
                self.register_or_append_data(save_name, data)

        if debug: print(f"Agent {component_name} outputs:\n{outputs}")
        if logging: self.logging_datapool()
        return outputs

    def _concurrent_run(
        self,
        pipeline: List[Union[str, Iterable[str]]],
        inputs: List[Tuple[str, Any]],
        logging: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ) -> Dict[str, List[Tuple[str, Any]]]:
        aggregated_outputs = defaultdict(list)
        for component_name in pipeline:
            if isinstance(component_name, list):
                outputs = self._recursive_run(
                    component_name, inputs, logging=logging, debug=debug, verbose=verbose
                )
            elif isinstance(component_name, tuple):
                outputs = self._concurrent_run(
                    component_name, inputs, logging=logging, debug=debug, verbose=verbose
                )
            else:
                outputs = self._run(
                    component_name, inputs, logging=logging, debug=debug, verbose=verbose
                )
            # Aggregate outputs
            aggregated_outputs[component_name].extend(outputs)
        return aggregated_outputs

    def _recursive_run(
        self,
        pipeline: List[Union[str, Iterable[str]]],
        inputs: List[Tuple[str, Any]],
        logging: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ) -> List[Tuple[str, Any]]:
        for component_name in pipeline:
            if isinstance(component_name, list):
                outputs = self._recursive_run(
                    component_name, inputs, logging=logging, debug=debug, verbose=verbose
                )
            elif isinstance(component_name, tuple):
                outputs = self._concurrent_run(
                    component_name, inputs, logging=logging, debug=debug, verbose=verbose
                    )
            else:
                outputs = self._run(
                    component_name, inputs, logging=logging, debug=debug, verbose=verbose
                )
            inputs = outputs
        return outputs

    def _recursive_name_check(
        self,
        pipeline: List[Union[str, Iterable[str]]],
    ) -> None:
        for component_name in pipeline:
            if isinstance(component_name, list):
                self._recursive_name_check(component_name)
            elif isinstance(component_name, tuple):
                self._recursive_name_check(component_name)
            else:
                assert component_name in self.agents or component_name in self.auxiliary_functions, f"Component {component_name} not found."

    def run(
        self,
        inputs: Union[Any, List[Any], Tuple[str, Any], List[Tuple[str, Any]]],
        pipeline: List[Union[str, Iterable[str]]] = None,
        datapool: Optional[Dict[str, Any]] = None,
        logging: bool = False,
        verbose: bool = False,
        debug: bool = False,
        **kwargs
    ) -> List[Tuple[str, Any]]:
        if pipeline is None:
            pipeline = self.pipeline

        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # Register auxiliary data to DataPool
        if datapool is not None:
            for key, value in datapool.items():
                self.register_data(key, value)
        # Check if all agent names are valid
        self._recursive_name_check(pipeline)
        # Run the pipeline
        outputs = self._recursive_run(
            pipeline, inputs, logging=logging, debug=debug, verbose=verbose
        )
        return outputs


if __name__ == "__main__":
    multi_agent = MultiAgent(
        agents={
            "singer": Agent(
                model="gemma3:1b-it-qat",
                base_url="http://localhost:8888/v1",
                system_prompt="You are a singer. Sing a song according to the request.",
                input_structure={"country": str, "year": int},
                output_structure={"song_name": str, "singer_name": str, "song_lyrics": str, "year": int},
            ),
            "guesser": Agent(
                model="gemma3:1b-it-qat",
                base_url="http://localhost:8888/v1",
                system_prompt="You are a music expert that knows all songs and singers. Given a song, guess the song and singer names.",
                input_structure={"song_lyrics": str, "hint": str},
                output_structure={"song_name": str, "singer_name": str, "singer_gender": Literal["male", "female"], "singer_nationality": str, "year": int},
            )
        },
        pipeline=[("singer", "singer", "singer"), "guesser"],
        output_to_register={
            "singer": {"song_name": "true_song_name"},
            "guesser": {"song_name": "pred_song_name"}
        },
        input_transformation={
            "guesser": lambda inputs, datapool: [(session_id, {"song_lyrics": input["song_lyrics"], "hint": datapool["hint"]}) for session_id, input in inputs["singer"]] # This must be batched processing
        },
        output_transformation={
            "guesser": lambda outputs, datapool: [(session_id, {"song_name": output["song_name"], "singer_name": output["singer_name"]}) for session_id, output in outputs]    # This must be batched processing
        }
    )
    # Single-turn interaction
    outputs: List[Tuple[str, Any]] = multi_agent.run(
        inputs={"country": "South Korea", "year": 2020},
        data_pool={"hint": "The song is about a singer"},
        debug=True,
    )
    print(outputs)
    print(multi_agent.datapool)