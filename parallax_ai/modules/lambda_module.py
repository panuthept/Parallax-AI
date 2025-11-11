from ..dataclasses import Job
from typing import Any, Callable
from dataclasses import dataclass
from .base_module import BaseModule


@dataclass
class LambdaModule(BaseModule):
    function: Callable[[dict], Any] = None

    def _create_job(self, instance_id: str, module_input: dict) -> Job:
        return Job(
            module_input=module_input,
            executor_func=self.function,
            executor_input=module_input,
            instance_id=instance_id,
            module_name=self.name,
            progress_name=self.progress_name
        )