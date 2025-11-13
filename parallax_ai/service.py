from .datapool import DataPool
from .modules import BaseModule
from .distributor import Distributor
from .composer import OutputComposer
from .dataclasses import Job, Instance
from typing import List, Dict, Optional


class Service:
    def __init__(
        self, 
        modules: List[BaseModule],
        name: str = None,
        datapool: Optional[DataPool] = None,
        worker_nodes: Optional[Dict[str, List[dict]]] = None,
        output_composers: Optional[List[OutputComposer]] = None,
        ray_remote_address: Optional[str] = None,
        ray_local_workers: Optional[int] = None,
        local_workers: Optional[int] = None,
        chunk_size: Optional[int] = 6000,
        **kwargs,
    ):
        self.name = name
        self.modules = modules
        self.flattened_modules = self.get_flattened_modules(modules)
        self.datapool = datapool if datapool is not None else DataPool()
        self.output_composers = output_composers
        self.worker_nodes = worker_nodes
        # Initialize Distributor
        self.distributor = Distributor(
            ray_remote_address=ray_remote_address,
            ray_local_workers=ray_local_workers,
            local_workers=local_workers,
            chunk_size=chunk_size,
            **kwargs
        )

    def update_worker_nodes(self, modules: List[BaseModule], worker_nodes: Optional[Dict[str, List[dict]]]):
        if worker_nodes is None:
            return
        # worker_nodes can be a path to JSON file or a dict
        if isinstance(worker_nodes, str):
            import json
            with open(worker_nodes, 'r') as f:
                worker_nodes_dict = json.load(f)
        else:
            worker_nodes_dict = worker_nodes
            
        for module in modules:
            module.worker_nodes = worker_nodes_dict

    @staticmethod
    def get_flattened_modules(modules: List[BaseModule]) -> List[BaseModule]:
        flattened_modules = []
        for module in modules:
            flattened_modules.extend(module.flatten())
        return flattened_modules

    def run(
        self, 
        inputs: List[dict] = None, 
        instances: List[Instance] = None,
        verbose: bool = True,
        debug_mode: bool = False,
    ) -> List[dict]:
        # Set worker nodes for all modules
        self.update_worker_nodes(self.flattened_modules, self.worker_nodes)

        # Add inputs to DataPool
        self.datapool.add(data=inputs, instances=instances)

        # Each module creates jobs based on available data in DataPool
        pendding_jobs: List[Job] = []
        for module in self.flattened_modules:
            for instance in self.datapool.retrieve(target_contents=module.interface.dependencies, avoid_modules=[module.name]):
                pendding_jobs.extend(module.create_jobs(instance))

        # Execute jobs
        completed_jobs = self.distributor(pendding_jobs, verbose=verbose, debug_mode=debug_mode)

        # Update DataPool
        for module in self.flattened_modules:
            jobs = [job for job in completed_jobs if job.module_name == module.name]
            for instand_id, new_contents in module.get_output_contents(jobs).items():
                self.datapool.update(instand_id, new_contents, executed_module=module.name)

        outputs = []
        # Get composed outputs if output_composers are provided
        if self.output_composers is not None:
            for composer in self.output_composers:
                for instance in self.datapool.retrieve(target_modules=composer.dependencies):
                    if composer.condition_satisfy(instance.contents):
                        composed_output = composer.compose(instance.contents)
                        outputs.append(composed_output)
                        # Clean up memory
                        self.datapool.remove(instance.id)
        # Get outputs from finished instances
        for instance in self.datapool.retrieve(target_modules=[module.name for module in self.flattened_modules]):
            output = instance.contents
            outputs.append(output)
            # Clean up memory
            self.datapool.remove(instance.id)
        return outputs