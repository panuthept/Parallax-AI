import os
import yaml
import json
from typing import *
from copy import deepcopy
from .agents.agent import Agent


class MultiAgent:
    def __init__(
        self,
        agents: Dict[str, Agent],
        logging_path: str = "./multiagent/logs.json",
    ):
        self.datapool = {}
        self.agents = agents
        self.logging_path = logging_path

    @classmethod
    def from_yaml(cls, path: str, **kwargs) -> "MultiAgent":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        agents = {}
        for name, config in data["agents"].items():
            if "input_structure" in config:
                config["input_structure"] = {k: eval(v) for k, v in config["input_structure"].items()}
            if "output_structure" in config:
                config["output_structure"] = {k: eval(v) for k, v in config["output_structure"].items()}
            if "input_transformation" in config:
                config["input_transformation"] = eval(config["input_transformation"])
            if "output_transformation" in config:
                config["output_transformation"] = eval(config["output_transformation"])
            agents[name] = Agent(**config)
        return cls(agents=agents, **kwargs)

    def register_data(self, data: Dict[str, Any]):
        self.datapool.update(deepcopy(data))

    def logging_datapool(self):
        os.makedirs(self.logging_path, exist_ok=True)
        with open(self.logging_path, "w") as f:
            json.dump(self.datapool, f)

    def run(
        self,
        pipeline: List[str],
        outputs: List[str] = None,
        logging: bool = True,
    ) -> Dict[str, Any]:
        for agent_name in pipeline:
            assert agent_name in self.agents, f"Agent {agent_name} not found"

        for agent_name in pipeline:
            outputs = self.agents[agent_name].run(self.datapool)
            self.datapool.update(outputs)
            if logging:
                self.logging_datapool()

        if outputs is None:
            return deepcopy(self.datapool)
        else:
            return {key: deepcopy(self.datapool[key]) for key in outputs}