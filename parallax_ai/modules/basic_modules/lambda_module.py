from typing import Any, Callable
from dataclasses import dataclass
from ...core.module import Module
from ...core.dataclasses import Job


@dataclass
class LambdaModule(Module):
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