from typing import Callable, Any
from dataclasses import dataclass


@dataclass
class Job:
    module_input: Any              # Inputs to the module

    executor_func: Callable         # Executor function
    executor_input: Any             # Input to be passed to the executor function
    executor_output: Any = None     # Output from the executor function
    execution_success: bool = False # Whether the execution was successful
    executor_default_output: Any = None

    instance_id: Any = None         # ID of the Instance this job is associated with
    module_name: str = None         # Module that created this job
    progress_name: str = None       # Name for progress bar

    def update_execution_result(self, output: Any, success: bool):
        if not success:
            output = self.executor_default_output
        self.executor_output = output
        self.execution_success = success