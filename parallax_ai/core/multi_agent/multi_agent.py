import os
import random
from copy import deepcopy
from ..agent import Agent
from ..client import Client
from types import GeneratorType
from ..engine import ParallaxEngine
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from .dataclasses import ModuleIO, AgentModule, Instance


class MultiAgent:
    def __init__(
        self,
        modules: Dict[str, AgentModule],
        client: Client = None,
        max_tries: int = 10,
        dismiss_none_output: bool = True,   # If True, None outputs from agents will be removed from outputs
    ):
        self.client = client
        self.modules = modules
        self._modules = self._flatten_modules(modules)
        self.max_tries = max_tries
        self.dismiss_none_output = dismiss_none_output
        self.engine = ParallaxEngine(
            client=self.client, max_tries=max_tries, dismiss_none_output=False
        )
        self.instances: Dict[str, Instance] = {}

    @property
    def leaf_modules(self) -> List[str]:
        """
        Get the names of modules that are not dependencies of any other modules.
        """
        dependent_modules = set()
        for module in self._modules.values():
            for dep in module.io.dependencies:
                if dep in self._modules:
                    dependent_modules.add(dep)
        leaf_modules = [name for name in self._modules.keys() if name not in dependent_modules]
        return leaf_modules
    
    @property
    def dependencies(self) -> Dict[str, List[str]]:
        """
        Get the dependencies of the this multi-agent system (what inputs needed from user).
        """
        dependencies = defaultdict(list)
        for agent_name, module in self._modules.items():
            for dep in module.io.dependencies:
                if dep not in self._modules:
                    dependencies[agent_name].append(dep)
        return dict(dependencies)

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
        agent_names = []
        for agent_name, agent_inputs in inputs.items():
            assert agent_name in self._modules, f"Agent {agent_name} not found."
            agent_jobs = self._modules[agent_name].agent._create_jobs(
                agent_inputs, progress_name=progress_names[agent_name] if progress_names is not None else None
            )
            jobs.extend(agent_jobs)
            agent_names.extend(agent_name for _ in range(len(agent_jobs)))
        if len(jobs) == 0:
            return [], [], []

        # Shuffle jobs to mix different agents' jobs
        indices = list(range(len(jobs)))
        random.shuffle(indices)
        shuffled_jobs = [jobs[i] for i in indices]
        shuffled_agent_names = [agent_names[i] for i in indices]

        # Process the jobs by the ParallaxEngine with retries
        shuffled_jobs = self.engine(shuffled_jobs, **kwargs)
        shuffled_outputs = [job.output for job in shuffled_jobs]

        # Update conversation memory with assistant outputs
        for job, agent_name in zip(shuffled_jobs, shuffled_agent_names):
            if job.output is not None:
                job.session_id = self._modules[agent_name].agent.conversation_memory.update_assistant(job.session_id, job.output)
        shuffled_session_ids = [job.session_id for job in shuffled_jobs]

        # Unshuffle jobs to the original order
        outputs = [None for _ in range(len(jobs))]
        session_ids = [None for _ in range(len(jobs))]
        agent_names = [None for _ in range(len(jobs))]
        for i, index in enumerate(indices):
            outputs[index] = shuffled_outputs[i]
            session_ids[index] = shuffled_session_ids[i]
            agent_names[index] = shuffled_agent_names[i]
        return agent_names, session_ids, outputs

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
        agent_names, session_ids, outputs = self.__run(inputs, progress_names=progress_names, **kwargs)
        
        # Get outputs for each agent
        agent_outputs = defaultdict(list)
        for agent_name, session_id, out in zip(agent_names, session_ids, outputs):
            if self._modules[agent_name].agent.conversational_agent:
                agent_outputs[agent_name].append((session_id, out))
            else:
                agent_outputs[agent_name].append(out)

        # Transform outputs for all agents
        for agent_name, outputs in agent_outputs.items():
            agent_outputs[agent_name] = self._modules[agent_name].agent.output_transformation(outputs)
        return agent_outputs
    
    def check_and_acquire_dependencies(self, contents: Dict[str, Any], dependencies: List[Any]) -> Dict[str, Any]:
        acquired_contents = {}
        for dep in dependencies:
            if dep in contents:
                acquired_contents[dep] = contents[dep]
                # Remove 'None' from agent outputs dependencies to improve quality of life
                if dep in self._modules:
                    assert isinstance(acquired_contents[dep], list), f"Agent '{dep}' outputs are supposed to be a list. But got {type(acquired_contents[dep])}."
                    acquired_contents[dep] = [out for out in acquired_contents[dep] if out is not None]
                    assert len(acquired_contents[dep]) > 0, f"All outputs from agent '{dep}' are None. Cannot proceed. This may be due to output formatting issues. Consider increasing max_tries or checking the agent's output format and system prompt."
            else:
                return None
        return acquired_contents
    
    def _get_pipeline_inputs(self):
        inputs = defaultdict(list)
        indexing = defaultdict(list)
        for agent_name, module in self._modules.items():
            for instance in self.instances.values():
                # Skip if this instance already has output for this agent
                if agent_name in instance.contents:
                    continue

                # Check and acquire dependencies
                dependencies = self.check_and_acquire_dependencies(instance.contents, module.io.dependencies)
                if dependencies is None:
                    continue

                if module.io.input_processing is not None:
                    agent_input = module.io.input_processing(dependencies)
                    if isinstance(agent_input, GeneratorType) or isinstance(agent_input, list):
                        for inp in agent_input:
                            inputs[agent_name].append(inp)
                            indexing[agent_name].append(instance.id)
                    else:
                        inputs[agent_name].append(agent_input)
                        indexing[agent_name].append(instance.id)
                else:
                    inputs[agent_name].append(dependencies)
                    indexing[agent_name].append(instance.id)
        return dict(inputs), dict(indexing)
    
    def _update_instances_with_outputs(self, inputs, outputs, indexing):
        for agent_name in outputs:
            assert len(inputs[agent_name]) == len(outputs[agent_name]), f"Number of inputs and outputs for agent {agent_name} do not match."
            agent_inputs = defaultdict(list)
            agent_outputs = defaultdict(list)
            for agent_input, agent_output, instance_id in zip(inputs[agent_name], outputs[agent_name], indexing[agent_name]):
                if self._modules[agent_name].agent.conversational_agent:
                    agent_output = agent_output[1]
                assert instance_id in self.instances, "Instance ID not found."
                agent_inputs[instance_id].append(agent_input)
                agent_outputs[instance_id].append(agent_output)
            # Update instance contents with agent outputs
            for instance_id in agent_outputs:
                agent_input = agent_inputs[instance_id]
                agent_output = agent_outputs[instance_id]
                if agent_output is None:
                    self.instances[instance_id].contents[agent_name] = None
                elif self._modules[agent_name].io.output_processing is not None:
                    # Process outputs if output_processing is provided
                    agent_output = self._modules[agent_name].io.output_processing(
                        deepcopy(agent_input),
                        deepcopy(agent_output),
                    )
                    assert isinstance(agent_output, (list, GeneratorType)) or agent_output is None, "output_processing must return a list or generator."
                    if isinstance(agent_output, GeneratorType):
                        self.instances[instance_id].contents[agent_name] = deepcopy(list(agent_output))
                    else:
                        self.instances[instance_id].contents[agent_name] = deepcopy(agent_output)
                else:
                    self.instances[instance_id].contents[agent_name] = deepcopy(agent_output)

    def _remove_none_outputs(self, outputs: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        cleaned_outputs = {}
        for agent_name, agent_outputs in outputs.items():
            if agent_name not in self._modules:
                continue
            cleaned_outputs[agent_name] = [out for out in agent_outputs if out is not None]
        return cleaned_outputs
    
    def run_single_step(
        self,
        inputs = None,
        verbose: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run a single step of the multi-agent pipeline.
        """
        # Create new package (if inputs is provided)
        if inputs is not None:
            for inp in inputs:
                assert isinstance(inp, dict), "Each input should be a dictionary of agent_name -> inputs."
                # Check if all required dependencies are provided
                for agent_name, deps in self.dependencies.items():
                    for dep in deps:
                        if dep not in inp:
                            raise ValueError(f"Input for dependency '{dep}' of agent '{agent_name}' is missing. Please provide all required inputs: {self.dependencies}")
                # Initialize instance
                instance = Instance(contents=inp)
                self.instances[instance.id] = instance
        elif len(self.instances) == 0:
            print("Warning: No packages to process. Please provide inputs or external_data to create a new package.")
            return []

        # Get inputs for all agents
        inputs, indexing = self._get_pipeline_inputs()

        # Execute Agents
        outputs = self._run(
            inputs=deepcopy(inputs), 
            progress_names={agent_name: module.progress_name for agent_name, module in self._modules.items()} if verbose else None,
            **kwargs
        )

        # Update instances with outputs
        self._update_instances_with_outputs(inputs, outputs, indexing)

        # Return done instances' contents and get the remaining instances
        return_contents = []
        remaining_instances = {}
        for i, instance in self.instances.items():
            if instance.is_completed(self.leaf_modules):
                if self.dismiss_none_output:
                    return_contents.append(self._remove_none_outputs(instance.contents))
                else:
                    return_contents.append(instance.contents)
            else:
                remaining_instances[i] = instance
        # Remove finished or stalled instances
        self.instances = remaining_instances
        return return_contents
    
    def flush(self, verbose: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """
        Finish all the remaining packages.
        """
        return_contents = []
        while len(self.instances) > 0:
            contents = self.run_single_step(verbose=verbose, **kwargs)
            if len(contents) > 0:
                return_contents.extend(contents)
        return return_contents
    
    def run(
        self,
        inputs,
        verbose: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run the given inputs through the multi-agent pipeline until all agents have produced outputs.
        """
        return_contents = self.run_single_step(inputs=inputs, verbose=verbose, **kwargs)
        return_contents.extend(self.flush(verbose=verbose, **kwargs))
        return return_contents