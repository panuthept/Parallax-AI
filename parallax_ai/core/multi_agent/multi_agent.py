import os
import random
from uuid import uuid4
from copy import deepcopy
from ..agent import Agent
from ..client import Client
from ..engine import ParallaxEngine
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from .dataclasses import ModuleIO, Package, Dependency, AgentModule


class MultiAgent:
    def __init__(
        self,
        modules: Dict[str, AgentModule],
        client: Client = None,
        max_tries: int = 5,
        dismiss_none_output: bool = True,
    ):
        self.client = client
        self.modules = modules
        self._modules = self._flatten_modules(modules)
        self.max_tries = max_tries
        self.dismiss_none_output = dismiss_none_output
        self.engine = ParallaxEngine(
            client=self.client, max_tries=max_tries, dismiss_none_output=False
        )
        self.packages: List[Package] = []

    def _flatten_modules(self, modules: Dict[str, AgentModule]) -> Dict[str, AgentModule]:
        """
        Breakdown AgentModule with multiple IOs into multiple AgentModules with single IO.
        Ex. {"agent1": AgentModule(agent=Agent1, io={"task1": IO1, "task2": IO2})}
            -> {"agent1.task1": AgentModule(agent=Agent1, io=IO1),
                "agent1.task2": AgentModule(agent=Agent1, io=IO2)}
        """
        flatten_modules = {}
        for agent_name, module in modules.items():
            if isinstance(module.io, dict):
                for io_name, io in module.io.items():
                    flatten_modules[f"{agent_name}.{io_name}"] = AgentModule(
                        agent=module.agent,
                        io=io,
                        progress_name=module.progress_name,
                    )
            else:
                flatten_modules[agent_name] = module
        return flatten_modules

    def save(self, name: str, cache_dir: str = "~/.cache/parallax_ai"):
        save_path = f"{cache_dir}/{name}"
        # Save Modules
        for agent_name, module in self.modules.items():
            module.agent.save(f"{save_path}/agents/{agent_name}.yaml")

        # Save IOs
        for agent_name, module in self.modules.items():
            if isinstance(module.io, dict):
                for io_name, io in module.io.items():
                    io.save(f"{save_path}/ios/{agent_name}.{io_name}.yaml")
            elif module.io is not None:
                module.io.save(f"{save_path}/ios/{agent_name}.yaml")

        # Save progress names
        progress_names = {}
        for agent_name, module in self.modules.items():
            if module.progress_name is not None:
                progress_names[agent_name] = module.progress_name
        with open(f"{save_path}/progress_names.yaml", "w") as f:
            import yaml
            yaml.dump(progress_names, f)

        # Save MultiAgent config
        config = {
            "modules": list(self.modules.keys()),
            "max_tries": self.max_tries,
            "dismiss_none_output": self.dismiss_none_output,
        }
        with open(f"{save_path}/multi_agent.yaml", "w") as f:
            import yaml
            yaml.dump(config, f)

    @classmethod
    def load(cls, name: str, cache_dir: str = "~/.cache/parallax_ai", client: Optional[Client] = None):
        load_path = f"{cache_dir}/{name}"
        # Load MultiAgent config
        with open(f"{load_path}/multi_agent.yaml", "r") as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Load Agents
        agents = {}
        for agent_name in config["modules"]:
            agents[agent_name] = Agent.load(f"{load_path}/agents/{agent_name}.yaml", client=client)
        
        # Load AgentIOs
        ios = {}
        for agent_name in config["modules"]:
            # IOs can be either single IO or multiple IOs
            if os.path.exists(f"{load_path}/ios/{agent_name}.yaml"):
                ios[agent_name] = ModuleIO.load(f"{load_path}/ios/{agent_name}.yaml")
            else:
                for filename in os.listdir(f"{load_path}/ios"):
                    if filename.startswith(f"{agent_name}.") and filename.endswith(".yaml"):
                        io_name = filename[len(agent_name)+1:-len(".yaml")]
                        if agent_name not in ios:
                            ios[agent_name] = {}
                        ios[agent_name][io_name] = ModuleIO.load(f"{load_path}/ios/{filename}")

        # Load progress names
        progress_names = {}
        if os.path.exists(f"{load_path}/progress_names.yaml"):
            with open(f"{load_path}/progress_names.yaml", "r") as f:
                import yaml
                progress_names = yaml.safe_load(f)

        modules = {
            agent_name: AgentModule(
                agent=agents[agent_name],
                io=ios.get(agent_name, None),
                progress_name=progress_names.get(agent_name, None),
            ) for agent_name in config["modules"]
        }
        
        return cls(
            modules=modules,
            max_tries=config.get("max_tries", 1),
            dismiss_none_output=config.get("dismiss_none_output", False),
            client=client,
        )

    def __run(
        self, 
        inputs: List[Tuple[str, Any]],
        progress_names: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Tuple[List[str], List[Any], List[str], List[str]]:
        # Create jobs for all agents
        jobs = []
        ori_inputs = []
        agent_names = []
        for agent_name, agent_inputs in inputs.items():
            assert agent_name in self._modules, f"Agent {agent_name} not found."
            agent_jobs = self._modules[agent_name].agent._create_jobs(
                agent_inputs, progress_name=agent_name if progress_names is None else progress_names[agent_name]
            )
            jobs.extend(agent_jobs)
            ori_inputs.extend(agent_inputs)
            agent_names.extend(agent_name for _ in range(len(agent_jobs)))
        if len(jobs) == 0:
            return [], [], [], []

        # Shuffle jobs to mix different agents' jobs
        indices = list(range(len(jobs)))
        random.shuffle(indices)
        shuffled_jobs = [jobs[i] for i in indices]
        shuffled_inputs = [ori_inputs[i] for i in indices]
        shuffled_agent_names = [agent_names[i] for i in indices]

        # Process the jobs by the ParallaxEngine with retries
        shuffled_jobs = self.engine(shuffled_jobs, **kwargs)
        # Get outputs in the original order
        # shuffled_inputs = [job.inp for job in shuffled_jobs]
        shuffled_outputs = [job.output for job in shuffled_jobs]

        # Update conversation memory with assistant outputs
        for job, agent_name in zip(shuffled_jobs, shuffled_agent_names):
            if job.output is not None:
                job.session_id = self._modules[agent_name].agent.conversation_memory.update_assistant(job.session_id, job.output)
        shuffled_session_ids = [job.session_id for job in shuffled_jobs]

        # Unshuffle jobs to the original order
        inputs = [None for _ in range(len(jobs))]
        outputs = [None for _ in range(len(jobs))]
        session_ids = [None for _ in range(len(jobs))]
        agent_names = [None for _ in range(len(jobs))]
        for i, index in enumerate(indices):
            inputs[index] = shuffled_inputs[i]
            outputs[index] = shuffled_outputs[i]
            session_ids[index] = shuffled_session_ids[i]
            agent_names[index] = shuffled_agent_names[i]

        # Dismiss None outputs (if dismiss_none_output is True)
        if self.dismiss_none_output:
            session_ids = [sid for sid, out in zip(session_ids, outputs) if out is not None]
            agent_names = [an for an, out in zip(agent_names, outputs) if out is not None]
            inputs = [inp for inp, out in zip(inputs, outputs) if out is not None]
            outputs = [out for out in outputs if out is not None]
        return agent_names, session_ids, inputs, outputs

    def _run(
        self, 
        inputs: Dict[str, Any], 
        progress_names: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        progress_names = deepcopy(progress_names)
                
        # Transform inputs for all agents
        for agent_name in inputs.keys():
            assert agent_name in self._modules, f"Agent {agent_name} not found."
            inputs[agent_name] = self._modules[agent_name].agent.input_transformation(inputs[agent_name])

        # Run all agents
        agent_names, session_ids, inputs, outputs = self.__run(inputs, progress_names=progress_names, **kwargs)
        
        # Get outputs for each agent
        agent_outputs = defaultdict(list)
        for agent_name, session_id, inp, out in zip(agent_names, session_ids, inputs, outputs):
            if self._modules[agent_name].agent.conversational_agent:
                agent_outputs[agent_name].append((session_id, inp, out))
            else:
                agent_outputs[agent_name].append((inp, out))

        # Transform outputs for all agents
        for agent_name, outputs in agent_outputs.items():
            agent_outputs[agent_name] = self._modules[agent_name].agent.output_transformation(outputs)
        return agent_outputs
    
    def init_package(
        self, 
        inputs: Optional[Dict[str, Any]] = None, 
        external_data: Optional[Dict[str, Any]] = None,
        tracking_id: Optional[str] = None,
    ) -> Package:
        package = Package(
            id=tracking_id if tracking_id is not None else uuid4().hex,
            external_data=external_data, 
        )
        if inputs is not None:
            for agent_name, agent_inputs in inputs.items():
                assert agent_name in self._modules, f"Unknown agent name: '{agent_name}' in inputs."
                package.agent_inputs[agent_name] = agent_inputs
        return package
    
    def is_dependency_fulfilled(self, package: Package, dependency: Dependency) -> bool:
        # Check agent_outputs dependencies
        if dependency.agent_outputs is not None:
            for output_name in dependency.agent_outputs:
                if output_name not in package.agent_outputs:
                    return False
        # Check external_data dependencies
        if dependency.external_data is not None:
            for data_name in dependency.external_data:
                if data_name not in package.external_data:
                    return False
        return True
    
    def _get_pipeline_inputs(
        self, 
        packages: List[Package], 
        modules: Optional[Dict[str, AgentModule]],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        inputs = {}
        package_indices = {}
        for i, package in enumerate(packages):   # Prioritize older packages
            if package is None or package.is_completed:
                continue
            # Get inputs from package.agent_inputs
            for agent_name, agent_inputs in package.agent_inputs.items():
                if agent_name in inputs:
                    # Already has this inputs
                    continue
                if agent_name in package.agent_outputs:
                    # Already has executed this agent
                    continue
                inputs[agent_name] = agent_inputs
                package_indices[agent_name] = i  # Record which package provides this input

            # Get inputs from package.external_data and package.agent_outputs
            for agent_name, module in modules.items():
                if agent_name in inputs:
                    # Already has this inputs
                    continue
                if agent_name in package.agent_outputs:
                    # Already has executed this agent
                    continue
                if module.io is None or module.io.input_processing is None:
                    # No input processing function
                    continue
                # Check dependencies and get relevant data
                relevance_agent_outputs = deepcopy(package.agent_outputs)
                relevance_external_data = deepcopy(package.external_data)
                if module.io.dependency is not None:
                    if not self.is_dependency_fulfilled(package, module.io.dependency):
                        # Dependencies not fulfilled
                        continue
                    # Filter only relevant agent outputs and external data
                    if module.io.dependency.agent_outputs is not None:
                        relevance_agent_outputs = {k: v for k, v in relevance_agent_outputs.items() if k in module.io.dependency.agent_outputs}
                    if module.io.dependency.external_data is not None:
                        relevance_external_data = {k: v for k, v in relevance_external_data.items() if k in module.io.dependency.external_data}
                # Get agent inputs
                agent_inputs = module.io.input_processing(relevance_agent_outputs, relevance_external_data)
                if agent_inputs is None:
                    print(f"[Warning] Obtain 'None' inputs for agent {agent_name}. This is possible if dependency is not provided for downstream Agent.")
                if len(agent_inputs) == 0:                    
                    print(f"[Warning] Obtain empty inputs for agent {agent_name}.")
                inputs[agent_name] = agent_inputs
                package_indices[agent_name] = i # Record which package provides this input
        return inputs, package_indices
    
    def _update_package_status(self, packages: List[Package], package_indices: Dict[str, int]):
        # (a package is finished if all agents have been executed)
        for i, package in enumerate(packages):
            if package is None or package.is_completed:
                continue
            all_agents_executed = True
            for agent_name in self._modules.keys():
                if agent_name not in package.agent_outputs:
                    all_agents_executed = False
            if all_agents_executed:
                package.is_completed = True
        # (a package is stalled if no new agents can be executed)
        for i, package in enumerate(packages):
            if package is None or package.is_completed:
                continue
            if i not in package_indices.values():
                package.is_completed = True
        return packages
    
    def run_single_step(
        self,
        inputs=None,
        tracking_id=None,
        external_data=None,
        **kwargs,
    ) -> List[Package]:
        """
        Run a single step of the multi-agent pipeline.
        """
        # Create new package (if inputs or external_data are provided)
        if inputs is not None or external_data is not None:
            package = self.init_package(inputs, external_data, tracking_id=tracking_id)
            self.packages.append(package)
        elif len(self.packages) == 0:
            print("Warning: No packages to process. Please provide inputs or external_data to create a new package.")
            return []

        # Input processing for all agents
        inputs, package_indices = self._get_pipeline_inputs(self.packages, self._modules)

        # Execute Agents
        input_outputs = self._run(
            inputs=deepcopy(inputs), 
            progress_names={agent_name: module.progress_name for agent_name, module in self._modules.items()},
            **kwargs
        )

        # Update packages with outputs
        for agent_name, agent_input_outputs in input_outputs.items():
            package_index = package_indices[agent_name]
            # Get inputs and outputs
            agent_inputs = []
            agent_outputs = []
            for agent_input_output in agent_input_outputs:
                if self._modules[agent_name].agent.conversational_agent:
                    agent_inputs.append(agent_input_output[1])
                    agent_outputs.append(agent_input_output[2])
                else:
                    agent_inputs.append(agent_input_output[0])
                    agent_outputs.append(agent_input_output[1])

            assert self.packages[package_index] is not None, "Package should not be None."
            # Process outputs if output_processing is provided
            if agent_name in self._modules and self._modules[agent_name].io is not None and self._modules[agent_name].io.output_processing is not None:
                agent_outputs = self._modules[agent_name].io.output_processing(
                    deepcopy(agent_inputs),                                 # inputs
                    deepcopy(agent_outputs),                                # outputs
                    deepcopy(self.packages[package_index].external_data)    # data
                )
            self.packages[package_index].agent_outputs[agent_name] = agent_outputs

        # Update package status
        self.packages = self._update_package_status(self.packages, package_indices)
        # Get return packages
        return_packages = [deepcopy(package) for package in self.packages if package is not None]
        # Remove finished or stalled packages
        self.packages = [package for package in self.packages if not package.is_completed]
        return return_packages
    
    def flush(self, **kwargs) -> List[Package]:
        """
        Finish all the remaining packages.
        """
        all_packages = {}
        while len(self.packages) > 0:
            packages = self.run_single_step(**kwargs)
            for package in packages:
                if package is not None:
                    all_packages[package.id] = package
        return list(all_packages.values())
    
    def run(
        self,
        inputs=None,
        external_data=None,
        **kwargs,
    ) -> List[Package]:
        """
        Run the given inputs through the multi-agent pipeline until all agents have produced outputs.
        """
        assert inputs is not None or external_data is not None, "Please provide inputs or external_data to run the multi-agent pipeline."
        packages = self.run_single_step(inputs=inputs, external_data=external_data, **kwargs)
        packages = {package.id: package for package in packages}
        for package in self.flush(**kwargs):
            packages[package.id] = package
        return list(packages.values())
    
    # def run_until(
    #     self,
    #     condition_fn,
    #     inputs=None,
    #     external_data=None,
    #     max_steps: int = 10,
    #     **kwargs,
    # ) -> List[Package]:
    #     """
    #     Run the multi-agent pipeline until the condition function is satisfied or max_steps is reached.
    #     The condition function takes in a dictionary of agent outputs and returns a boolean.
    #     """
    #     assert inputs is not None or external_data is not None, "Please provide inputs or external_data to run the multi-agent pipeline."
    #     steps = 0
    #     packages = self.run_single_step(inputs=inputs, external_data=external_data, **kwargs)
    #     packages = {package.id: package for package in packages}
    #     while not condition_fn(outputs) and steps < max_steps:
    #         for package in self.run_single_step(**kwargs):
    #             packages[package.id] = package
    #         steps += 1
    #     return list(packages.values())