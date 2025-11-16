from typing import Dict
from ..dataclasses import Job
from dataclasses import dataclass
from .base_module import BaseModule


@dataclass
class SwitchModule(BaseModule):
    condition_key: str = None
    cases: Dict[str, BaseModule] = None

    def _create_job(self, instance_id: str, module_input: dict) -> Job:
        assert self.condition_key in module_input, \
            f"Condition key '{self.condition_key}' not found in module input."
        condition_value = module_input[self.condition_key]

        assert condition_value in self.cases, \
            f"Condition value '{condition_value}' not found in cases."
        selected_module = self.cases[condition_value]
        selected_module.worker_nodes = self.worker_nodes
        
        job = selected_module._create_job(instance_id, module_input)
        job.module_name = self.name  # Keep the SwitchModule's name
        return job