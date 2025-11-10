from copy import deepcopy
from abc import abstractmethod
from dataclasses import dataclass
from ..dataclasses import Job, Instance
from .module_interface import ModuleInterface
from typing import Optional, Union, Dict, List


@dataclass
class BaseModule:
    name: str = None
    interface: Optional[Union[ModuleInterface, Dict[str, ModuleInterface]]] = None
    worker_nodes: Optional[Dict[str, List[dict]]] = None
    progress_name: Optional[str] = None

    @abstractmethod
    def _create_job(self, instance_id: str, module_input: dict) -> Job:
        raise NotImplementedError("Subclasses must implement _create_job method.")
    
    def create_jobs(self, instance: Instance) -> List[Job]:
        module_inputs = self.interface.get_module_inputs(instance)
        jobs = [self._create_job(instance.id, module_input) for module_input in module_inputs]
        return jobs
    
    def get_output_contents(self, jobs: List[Job]) -> dict:
        instance_ids = set()
        instance_inputs = {}
        instance_outputs = {}
        # Get inputs and outputs
        for job in jobs:
            instance_ids.add(job.instance_id)

            if job.instance_id not in instance_inputs:
                instance_inputs[job.instance_id] = []
            instance_inputs[job.instance_id].append(job.module_input)
            
            if job.instance_id not in instance_outputs:
                instance_outputs[job.instance_id] = []
            instance_outputs[job.instance_id].append(job.executor_output)
        # Get module outputs
        module_outputs = {}
        for instance_id in instance_ids:
            if self.interface is not None and self.interface.output_processing is not None:
                outputs = self.interface.get_module_outputs(
                    inputs=instance_inputs[instance_id],
                    outputs=instance_outputs[instance_id]
                )
            else:
                outputs = {self.name: instance_outputs[instance_id]}
            module_outputs[instance_id] = outputs
        return module_outputs
    
    def flatten(self) -> List['BaseModule']:
        """
        Breakdown Module with multiple IOs into multiple Module with single IO.
        Ex. Module(name="module", ... io={"task1": ModuleIO, "task2": ModuleIO})
            -> [Module(name="module.task1", ... io=ModuleIO),
                Module(name="module.task2", ... io=ModuleIO)]
        """
        flatten_modules = []
        if self.interface is not None and isinstance(self.interface, dict):
            for io_name, io in self.interface.items():
                copied_module = deepcopy(self)
                copied_module.name = f"{self.name}.{io_name}"
                copied_module.interface = deepcopy(io)
                flatten_modules.append(copied_module)
        else:
            flatten_modules.append(self)
        return flatten_modules
    
    def run(
        self, 
        inputs: List[dict] = None, 
        instances: List[Instance] = None,
        debug_mode: bool = False,
        verbose: bool = True
    ) -> List[dict]:
        from ..service import Service

        # Create a temporary Service to run this module
        temp_service = Service(
            modules=[self],
            datapool=None,
            output_composers=None,
            worker_nodes=self.worker_nodes,
            debug_mode=debug_mode,
        )

        # Run the temporary Service
        outputs = temp_service.run(
            inputs=inputs,
            instances=instances,
            verbose=verbose
        )

        return outputs