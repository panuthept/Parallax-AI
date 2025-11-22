import numpy as np
from ...dataclasses import Job
from typing import List, Optional
from dataclasses import dataclass, field
from ...utilities import get_dummy_output
from ...modules.base_module import BaseModule
from ..agent_modules.agent_module import ModelSpec, auto_completions


def safeguard_completions(inputs: dict) -> dict:
    _, logprobs = auto_completions(inputs, return_logprobs=True)
    label_logprobs = [(inputs["representative_tokens"][token], logprob) for token, logprob in logprobs[inputs["representative_token_index"]] if token in inputs["representative_tokens"]]
    
    logprobs = [logprob for label, logprob in label_logprobs]
    labels = [label for label, logprob in label_logprobs]
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    class_probs = [(label, prob.item()) for label, prob in zip(labels, probs)]

    harmful_score = 0.0
    for label, prob in class_probs:
        if label.lower() in ["unsafe", "harmful"]:
            harmful_score += prob
        elif label.lower() in ["sensitive"]:
            harmful_score += prob * 0.5
    return {"harmful_score": harmful_score}

@dataclass
class BaseGuardModule(BaseModule):
    spec: ModelSpec = None
    max_retries: int = 10000000000
    representative_token_index: int = 0
    representative_tokens: dict = field(default_factory=lambda: 
        {
            "safe": "Safe", 
            "unsafe": "Harmful"
        }
    )

    @property
    def dependencies(self) -> List[str]:
        return ["prompt"]

    @property
    def input_structure(self) -> dict:
        return {"prompt": str, "response": Optional[str]}
    
    @property
    def output_structure(self) -> dict:
        return {"harmful_score": float}

    def get_safeguard_input(self, module_input: dict) -> dict:
        task = "prompt_classification"
        messages = [{"role": "user", "content": module_input["prompt"]}] 
        if module_input.get("response") is not None:
            task = "response_classification"
            messages.append({"role": "assistant", "content": module_input["response"]})
        return {"messages": messages, "task": task}

    def get_executor_input(self, module_input: dict) -> dict:
        assert self.worker_nodes is not None, "worker_nodes must be provided for AgentModule."
        assert self.spec.model_name in self.worker_nodes, f"Model addresses for model '{self.spec.model_name}' not found in worker_nodes."

        safeguard_input = self.get_safeguard_input(module_input)
        executor_input = {
            **safeguard_input,
            "representative_tokens": self.representative_tokens,
            "representative_token_index": self.representative_token_index,
            "model": self.spec.model_name,
            "model_addresses": self.worker_nodes[self.spec.model_name],
            "max_retries": self.max_retries,
            "kwargs": {
                "max_tokens": 100,
                "logprobs": True,
                "top_logprobs": 20,
            }
        }
        return executor_input

    def _create_job(self, instance_id: str, module_input: dict) -> Job:
        return Job(
            module_input=module_input,
            executor_func=safeguard_completions,
            executor_input=self.get_executor_input(module_input),
            executor_default_output=get_dummy_output(self.output_structure, default_value=1/len(self.representative_tokens)),
            instance_id=instance_id,
            module_name=self.name,
            progress_name=self.progress_name
        )