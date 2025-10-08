import random
from uuid import uuid4
from ..agent import Agent
from copy import deepcopy
from ..client import Client
from collections import defaultdict
from ..engine import ParallaxEngine
from typing import Any, Dict, List, Optional, Tuple
from .dataclasses import AgentIO, Package, Dependency


class MultiAgent:
    def __init__(
        self, 
        agents: Dict[str, Agent],
        agent_ios: Optional[Dict[str, AgentIO]] = None,
        progress_names: Optional[Dict[str, str]] = None,
        client: Optional[Client] = None,
        max_tries: int = 1,
        dismiss_none_output: bool = False,
    ):
        self.agents = agents
        self.agent_ios = agent_ios if agent_ios is not None else {}
        self.progress_names = progress_names if progress_names is not None else {}
        self.max_tries = max_tries
        self.dismiss_none_output = dismiss_none_output
        self.client = client if client is not None else self.agents[list(agents.keys())[0]].client
        self.engine = ParallaxEngine(
            client=self.client, max_tries=max_tries, dismiss_none_output=False
        )

        for name, agent in self.agents.items():
            if agent.max_tries != max_tries:
                print(f"Warning: Agent '{name}' has max_tries={agent.max_tries}, but ParallaxMultiAgent has max_tries={max_tries}. Overriding agent's setting.")
            if agent.dismiss_none_output != dismiss_none_output:
                print(f"Warning: Agent '{name}' has dismiss_none_output={agent.dismiss_none_output}, but ParallaxMultiAgent has dismiss_none_output={dismiss_none_output}. Overriding agent's setting.")
        
        self.packages: List[Package] = []

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
            assert agent_name in self.agents, f"Agent {agent_name} not found."
            agent_jobs = self.agents[agent_name]._create_jobs(
                agent_inputs, progress_name=agent_name if progress_names is None else progress_names[agent_name]
            )
            jobs.extend(agent_jobs)
            agent_names.extend(agent_name for _ in range(len(agent_jobs)))
        if len(jobs) == 0:
            return [], [], [], []

        # Shuffle jobs to mix different agents' jobs
        indices = list(range(len(jobs)))
        random.shuffle(indices)
        shuffled_jobs = [jobs[i] for i in indices]
        shuffled_agent_names = [agent_names[i] for i in indices]

        # Process the jobs by the ParallaxEngine with retries
        shuffled_jobs = self.engine(shuffled_jobs, **kwargs)
        # Get outputs in the original order
        shuffled_inputs = [job.inp for job in shuffled_jobs]
        shuffled_outputs = [job.output for job in shuffled_jobs]
        # Update conversation memory with assistant outputs
        for job, agent_name in zip(shuffled_jobs, shuffled_agent_names):
            if job.output is not None:
                job.session_id = self.agents[agent_name].conversation_memory.update_assistant(job.session_id, job.output)
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
        return_inputs: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        # Use default progress names if not provided
        if progress_names is None:
            progress_names = deepcopy(self.progress_names)
        for agent_name in inputs.keys():
            if agent_name not in progress_names:
                progress_names[agent_name] = self.progress_names.get(agent_name, None)
                
        # Transform inputs for all agents
        for agent_name in inputs.keys():
            assert agent_name in self.agents, f"Agent {agent_name} not found."
            inputs[agent_name], progress_names[agent_name] = self.agents[agent_name].input_transformation(inputs[agent_name], progress_names.get(agent_name, None))

        # Run all agents
        agent_names, session_ids, inputs, outputs = self.__run(inputs, progress_names=progress_names, **kwargs)
        
        # Get outputs for each agent
        dict_outputs = defaultdict(list)
        for agent_name, session_id, output in zip(agent_names, session_ids, outputs):
            if self.agents[agent_name].conversational_agent:
                if return_inputs:
                    dict_outputs[agent_name].append((session_id, inputs, output))
                else:
                    dict_outputs[agent_name].append((session_id, output))
            else:
                if return_inputs:
                    dict_outputs[agent_name].append((inputs, output))
                else:
                    dict_outputs[agent_name].append(output)

        # Transform outputs for all agents
        for agent_name in dict_outputs.keys():
            dict_outputs[agent_name] = self.agents[agent_name].output_transformation(dict_outputs[agent_name])
        return dict_outputs
    
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
                assert agent_name in self.agents, f"Unknown agent name: '{agent_name}' in inputs."
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
        agent_ios: Optional[Dict[str, AgentIO]],
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        inputs = {}
        package_indices = {}
        for i, package in enumerate(packages):   # Prioritize older packages
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

            # Get inputs from package.external_data
            for agent_name, agent_io in agent_ios.items():
                if agent_name in inputs:
                    # Already has this inputs
                    continue
                if agent_name in package.agent_outputs:
                    # Already has executed this agent
                    continue
                if agent_io.input_processing is None:
                    # No input processing function
                    continue
                if agent_io.dependency is not None:
                    if not self.is_dependency_fulfilled(package, agent_io.dependency):
                        # Dependencies not fulfilled
                        continue
                # Get agent inputs
                agent_inputs = agent_io.input_processing(deepcopy(package.agent_outputs), deepcopy(package.external_data))
                if agent_inputs is None:
                    print(f"[Warning] Obtain 'None' inputs for agent {agent_name}. This is normal if dependency is not provided for AgentIO.")
                if len(agent_inputs) == 0:                    
                    print(f"[Warning] Obtain empty inputs for agent {agent_name}.")
                inputs[agent_name] = agent_inputs
                package_indices[agent_name] = i # Record which package provides this input
        return inputs, package_indices
    
    def _clear_packages(self, packages: List[Package], package_indices: Dict[str, int]):
        removable_package_indices = set()
        # (a package is finished if all agents have been executed)
        for i, package in enumerate(packages):
            all_agents_executed = True
            for agent_name in self.agents.keys():
                if agent_name not in package.agent_outputs:
                    all_agents_executed = False
            if all_agents_executed:
                removable_package_indices.add(i)
        # (a package is stalled if no new agents can be executed)
        for i, package in enumerate(packages):
            if i not in package_indices.values():
                removable_package_indices.add(i)
        # Remove packages
        packages = [package for i, package in enumerate(packages) if i not in removable_package_indices]
        return packages
    
    def run_single_step(
        self,
        inputs=None,
        tracking_id=None,
        external_data=None,
        return_tracking_id=False,
        **kwargs,
    ):
        """
        Run a single step of the multi-agent pipeline.
        """
        # Create new package (if inputs or external_data are provided)
        if inputs is not None or external_data is not None:
            package = self.init_package(inputs, external_data, tracking_id=tracking_id)
            self.packages.append(package)
        else:
            if len(self.packages) == 0:
                print("Warning: No packages to process. Please provide inputs or external_data to create a new package.")
                return {}

        # Input processing for all agents
        inputs, package_indices = self._get_pipeline_inputs(self.packages, self.agent_ios)

        # Execute Agents
        input_outputs = self._run(
            inputs=deepcopy(inputs), return_inputs=True, **kwargs
        )

        # Update packages with outputs
        for agent_name, agent_input_outputs in input_outputs.items():
            package_index = package_indices[agent_name]
            # Process outputs if output_processing is provided
            if agent_name in self.agent_ios and self.agent_ios[agent_name].output_processing is not None:
                agent_inputs = agent_input_outputs[1] if self.agents[agent_name].conversational_agent else agent_input_outputs[0]
                agent_outputs = agent_input_outputs[-1]
                agent_outputs = self.agent_ios[agent_name].output_processing(
                    deepcopy(agent_inputs),                                 # inputs
                    deepcopy(agent_outputs),                                # outputs
                    deepcopy(self.packages[package_index].external_data)    # data
                )
            self.packages[package_index].agent_outputs[agent_name] = agent_outputs

        # Get return all outputs
        # [NOTE] Do not move this line after clearing packages
        if return_tracking_id:
            all_outputs = [(pkg.id, pkg.agent_outputs) for pkg in self.packages]
        else:
            all_outputs = [pkg.agent_outputs for pkg in self.packages]

        # Remove finished or stalled packages
        self.packages = self._clear_packages(self.packages, package_indices)

        return all_outputs
    
    def flush(self, **kwargs):
        """
        Finish all the remaining packages.
        """
        all_outputs = {}
        while len(self.packages) > 0:
            outputs = self.run_single_step(return_tracking_id=True, **kwargs)
            for id, out in outputs:
                all_outputs[id] = out
        return list(all_outputs.values())
    
    def run(
        self,
        inputs=None,
        external_data=None,
        **kwargs,
    ):
        """
        Run the given inputs through the multi-agent pipeline until all agents have produced outputs.
        """
        assert inputs is not None or external_data is not None, "Please provide inputs or external_data to run the multi-agent pipeline."
        outputs = self.run_single_step(inputs=inputs, external_data=external_data, **kwargs)[0]
        flush_outputs = self.flush(**kwargs)
        if len(flush_outputs) > 0:
            outputs = flush_outputs[0]
        return outputs
    
    def run_until(
        self,
        condition_fn,
        inputs=None,
        external_data=None,
        max_steps: int = 10,
        **kwargs,
    ):
        """
        Run the multi-agent pipeline until the condition function is satisfied or max_steps is reached.
        The condition function takes in a dictionary of agent outputs and returns a boolean.
        """
        assert inputs is not None or external_data is not None, "Please provide inputs or external_data to run the multi-agent pipeline."
        steps = 0
        outputs = self.run_single_step(inputs=inputs, external_data=external_data, **kwargs)[0]
        while not condition_fn(outputs) and steps < max_steps:
            all_outputs = self.run_single_step(**kwargs)
            if len(all_outputs) > 0:
                outputs = all_outputs[0]
            steps += 1
        return outputs