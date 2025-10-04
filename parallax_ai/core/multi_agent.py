import random
from .agent import Agent
from .client import ParallaxClient
from collections import defaultdict
from .engine import ParallaxEngine
from typing import Any, Dict, List, Optional, Tuple


class ParallaxMultiAgent:
    def __init__(
        self, 
        agents: Dict[str, Agent],
        client: Optional[ParallaxClient] = None,
        max_tries: int = 5,
    ):
        self.agents = agents
        self.max_tries = max_tries
        self.client = client if client is not None else self.agents[list(agents.keys())[0]].client
        self.engine = ParallaxEngine(
            client=self.client, max_tries=max_tries
        )

    def _run(
        self, 
        inputs: List[Tuple[str, Any]],
        progress_names: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Tuple[List[str], List[Any], List[str]]:
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
            return [], [], []

        # Shuffle jobs to mix different agents' jobs
        indices = list(range(len(jobs)))
        random.shuffle(indices)
        shuffled_jobs = [jobs[i] for i in indices]
        shuffled_agent_names = [agent_names[i] for i in indices]

        # Process the jobs by the ParallaxEngine with retries
        shuffled_jobs = self.engine(shuffled_jobs, **kwargs)
        # Get outputs in the original order
        shuffled_outputs = [job.output for job in shuffled_jobs]
        # Update conversation memory with assistant outputs
        for job, agent_name in zip(shuffled_jobs, shuffled_agent_names):
            if job.output is not None:
                job.session_id = self.agents[agent_name].conversation_memory.update_assistant(job.session_id, job.output)
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

    def run(
        self, 
        inputs: Dict[str, Any], 
        progress_names: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if progress_names is None:
            progress_names = {}
        # Transform inputs for all agents
        for agent_name in inputs.keys():
            assert agent_name in self.agents, f"Agent {agent_name} not found."
            inputs[agent_name], progress_names[agent_name] = self.agents[agent_name].input_transformation(inputs[agent_name], progress_names.get(agent_name, None))

        # Run all agents
        agent_names, session_ids, outputs = self._run(inputs, progress_names=progress_names, **kwargs)
        
        # Get outputs for each agent
        outputs = defaultdict(list)
        for agent_name, session_id, output in zip(agent_names, session_ids, outputs):
            if self.agents[agent_name].conversational_agent:
                outputs[agent_name].append((session_id, output))
            else:
                outputs[agent_name].append(output)

        # Transform outputs for all agents
        for agent_name in outputs.keys():
            outputs[agent_name] = self.agents[agent_name].output_transformation(outputs[agent_name])
        return outputs