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
        max_tries: int = 5,
        dismiss_none_output: bool = True,   # Not implemented yet
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
                agent_inputs, progress_name=agent_name if progress_names is None else progress_names[agent_name]
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
    
    def is_dependency_fulfilled(self, contents: Dict[str, Any], dependencies: List[str]) -> bool:
        fullfilled = True
        for dep in dependencies:
            if dep not in contents:
                fullfilled = False
                break
        return fullfilled
    
    def _get_pipeline_inputs(self):
        inputs = defaultdict(list)
        indexing = defaultdict(list)
        for agent_name, module in self._modules.items():
            for instance in self.instances.values():
                for content_node in instance.content_nodes.values():
                    # Node will be skipped under two conditions:
                    # 1. This agent has already been executed in this node
                    # 2. Dependencies are not fulfilled
                    # 3. Content node has no contents (happens when previous agent output is None)
                    if content_node.contents is None or len(content_node.contents) == 0:
                        continue
                    # Check if this agent has already been executed in this node
                    if agent_name in content_node.child_nodes:
                        continue
                    # Check if dependencies are fulfilled
                    if module.io is None:
                        # If IO is not defined, pass everything in the node's contents as inputs
                        inputs[agent_name].append(deepcopy(content_node.contents))
                        indexing[agent_name].append((instance.id, content_node.id))
                    else:
                        # Check if dependencies is fullfilled andfFilter only relevant contents based on dependencies
                        agent_input = deepcopy(content_node.contents)
                        if module.io.dependencies is not None:
                            if not self.is_dependency_fulfilled(content_node.contents, module.io.dependencies):
                                continue
                            # Filter only relevant contents
                            agent_input = {k: v for k, v in agent_input.items() if k in module.io.dependencies}
                        # Process inputs if input_processing is provided
                        if module.io.input_processing is not None:
                            agent_input = module.io.input_processing(agent_input)
                            if isinstance(agent_input, GeneratorType):
                                for inp in agent_input:
                                    inputs[agent_name].append(inp)
                                    indexing[agent_name].append((instance.id, content_node.id))
                            else:
                                inputs[agent_name].append(agent_input)
                                indexing[agent_name].append((instance.id, content_node.id))
                        else:
                            # Get agent inputs
                            inputs[agent_name].append(agent_input)
                            indexing[agent_name].append((instance.id, content_node.id))
        return inputs, indexing
    
    def _update_instances_with_outputs(self, inputs, outputs, indexing):
        for agent_name in outputs:
            assert len(inputs[agent_name]) == len(outputs[agent_name]), f"Number of inputs and outputs for agent {agent_name} do not match."
            # NOTE: This implementation assumes that each agent produces outputs for each input in order.
            for agent_input, agent_output, (instance_id, node_id) in zip(inputs[agent_name], outputs[agent_name], indexing[agent_name]):
                if self._modules[agent_name].agent.conversational_agent:
                    agent_output = agent_output[1]
                assert instance_id in self.instances, "Instance ID not found."
                assert node_id in self.instances[instance_id].content_nodes, "Content Node ID not found."
                # Update instance contents with agent outputs
                # Process outputs if output_processing is provided
                if agent_output is not None and self._modules[agent_name].io is not None and self._modules[agent_name].io.output_processing is not None:
                    agent_output = self._modules[agent_name].io.output_processing(
                        deepcopy(agent_input),
                        deepcopy(agent_output),
                    )
                    if isinstance(agent_output, GeneratorType):
                        for out in agent_output:
                            self.instances[instance_id].add_content_node(
                                parent_node_id=node_id,
                                agent_name=agent_name,
                                contents={agent_name: deepcopy(out)},
                            )
                    else:
                        self.instances[instance_id].add_content_node(
                            parent_node_id=node_id,
                            agent_name=agent_name,
                            contents={agent_name: deepcopy(agent_output)},
                        )
                else:
                    self.instances[instance_id].add_content_node(
                        parent_node_id=node_id,
                        agent_name=agent_name,
                        contents={agent_name: deepcopy(agent_output)},
                    )
    
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
                instance = Instance(inp)
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

        # Return done instances' contents
        return_contents = []
        for instance in self.instances.values():
            return_contents.append(instance.contents)
        # Remove finished or stalled instances
        self.instances = {i: instance for i, instance in self.instances.items() if not instance.is_completed(list(self._modules.keys()))}
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