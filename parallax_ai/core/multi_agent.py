import os
import yaml
import json
from typing import *
from copy import deepcopy
from typing_validation import validate
from parallax_ai.core.agents.agent import Agent


class DataPool:
    def __init__(self):
        self.data: Dict[str, Any] = {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(data=data)

    def to_dict(self) -> Dict[str, Any]:
        return deepcopy(self.data)

    def update(self, data: Dict[str, Any]):
        self.data.update(data)

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def search(self, structure: Dict[str, type]) -> Any:
        matched_data = {}
        for key, value in structure.items():
            if key in self.data:
                try:
                    validate(self.data[key], value)
                except TypeError:
                    continue
                matched_data[key] = self.data[key]
        
        # Check if all matched data have the same length
        if len(set(map(len, matched_data.values()))) != 1:
            raise ValueError("All matched data must have the same length")
        
        # Convert matched data to list of dict
        outputs = [{k: v[i] for k, v in matched_data.items()} for i in range(len(matched_data[key]))]
        return outputs


class MultiAgent:
    def __init__(
        self,
        agents: Dict[str, Agent],
        pipeline: List[str|Iterable[str]] = None,
        pool_data: Optional[Dict[str, Dict[str, str]]] = None,
        input_transformation: Optional[Dict[str, Callable]] = None,
        output_transformation: Optional[Dict[str, Callable]] = None,
        logging_path: str = "./multiagent/logs.json",
    ):
        self.datapool = {}
        self.agents = agents
        self.pipeline = pipeline
        self.pool_data = pool_data if pool_data is not None else {}
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
        agent_name: str, 
        inputs: List[Any], 
        logging: bool = False, 
        debug: bool = False
    ) -> List[Any]:
        # Input transformation
        if agent_name in self.input_transformation:
            inputs = self.input_transformation[agent_name](inputs, self.datapool)
        # Register inputs to datapool
        self.register_or_append_data(agent_name + "_inputs", inputs)

        outputs = self.agents[agent_name].run(inputs)

        # Output transformation
        if agent_name in self.output_transformation:
            outputs = self.output_transformation[agent_name](outputs, self.datapool)
        # Register outputs to datapool
        self.register_or_append_data(agent_name + "_outputs", outputs)
        # Update pool data
        if agent_name in self.pool_data:
            for key_name, save_name in self.pool_data[agent_name].items():
                data = []
                for output in outputs:
                    if output is None:
                        continue
                    assert key_name in output, f"Key '{key_name}' not found in the outputs of the agent named: {agent_name}"
                    data.append(output[key_name])
                self.register_or_append_data(save_name, data)

        if debug: print(f"Agent {agent_name} outputs:\n{outputs}")
        if logging: self.logging_datapool()
        return outputs

    def _concurrent_run(
        self,
        pipeline: List[str|Iterable[str]],
        inputs: List[Any],
        logging: bool = False,
        debug: bool = False,
    ) -> List[Any]:
        aggregated_outputs = []
        for agent_name in pipeline:
            if isinstance(agent_name, list):
                outputs = self._recursive_run(agent_name, inputs, logging, debug)
            elif isinstance(agent_name, tuple):
                outputs = self._concurrent_run(agent_name, inputs, logging, debug)
            else:
                outputs = self._run(agent_name, inputs, logging, debug)
            aggregated_outputs.append(outputs)
        return aggregated_outputs

    def _recursive_run(
        self,
        pipeline: List[str|Iterable[str]],
        inputs: List[Any],
        logging: bool = False,
        debug: bool = False,
    ) -> List[Any]:
        for agent_name in pipeline:
            if isinstance(agent_name, list):
                outputs = self._recursive_run(agent_name, inputs, logging, debug)
            elif isinstance(agent_name, tuple):
                outputs = self._concurrent_run(agent_name, inputs, logging, debug)
            else:
                outputs = self._run(agent_name, inputs, logging, debug)
            inputs = outputs
        return outputs

    def _recursive_name_check(
        self,
        pipeline: List[str|Iterable[str]],
    ) -> None:
        for agent_name in pipeline:
            if isinstance(agent_name, list):
                self._recursive_name_check(agent_name)
            elif isinstance(agent_name, tuple):
                self._recursive_name_check(agent_name)
            else:
                assert agent_name in self.agents, f"Agent {agent_name} not found"

    def run(
        self,
        inputs: Any,
        pipeline: List[str|Iterable[str]] = None,
        auxiliary_data: Optional[Dict[str, Any]] = None,
        logging: bool = False,
        debug: bool = False,
    ) -> List[Any]:
        if pipeline is None:
            pipeline = self.pipeline
        if not isinstance(inputs, list):
            inputs = [inputs]
        # Register auxiliary data to DataPool
        if auxiliary_data is not None:
            for key, value in auxiliary_data.items():
                self.register_data(key, value)
        # Check if all agent names are valid
        self._recursive_name_check(pipeline)
        # Run the pipeline
        outputs = self._recursive_run(pipeline, inputs, logging, debug)
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
        pool_data={
            "singer": {"song_name": "true_song_name"},
            "guesser": {"song_name": "pred_song_name"}
        },
        input_transformation={
            "guesser": lambda inputs, datapool: [{"song_lyrics": input["song_lyrics"], "hint": datapool["hint"]} for input in inputs] # This must be batched processing
        },
        output_transformation={
            "guesser": lambda outputs, datapool: [{"song_name": output["song_name"], "singer_name": output["singer_name"]} for output in outputs]    # This must be batched processing
        }
    )
    outputs = multi_agent.run(
        inputs={"country": "South Korea", "year": 2020},
        pipeline=["singer", "guesser"],
        auxiliary_data={"hint": "The song is about a singer"},
        debug=True,
    )
    print(outputs)
    print(multi_agent.datapool)
    