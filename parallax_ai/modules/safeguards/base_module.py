import numpy as np
from ...dataclasses import Job
from typing import List, Optional
from ..base_module import BaseModule
from dataclasses import dataclass, field
from ...utilities import get_dummy_output
from ..agent_module import ModelSpec, chat_completions, prompt_completions


def safeguard_completions(inputs: dict) -> dict:
    if "prompt" in inputs:
        output = prompt_completions(inputs)
        label_logprobs = [(inputs["representative_tokens"][token], logprob) for token, logprob in output.choices[0].logprobs.top_logprobs[inputs["representative_token_index"]].items() if token in inputs["representative_tokens"]]
    elif "messages" in inputs:
        output = chat_completions(inputs)
        label_logprobs = [[(top_logprob.token, top_logprob.logprob) for top_logprob in content.top_logprobs if top_logprob.token] for content in output.choices[0].logprobs.content]
        label_logprobs = [(inputs["representative_tokens"][token], logprob) for token, logprob in label_logprobs[inputs["representative_token_index"]] if token in inputs["representative_tokens"]]
    logprobs = [logprob for label, logprob in label_logprobs]
    labels = [label for label, logprob in label_logprobs]
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    class_probs = [(label, prob.item()) for label, prob in zip(labels, probs)]

    harmful_score = 0.0
    for label, prob in class_probs:
        if inputs.get("task") == "prompt_classification":
            if label.lower() in ["unsafe", "harmful", "sensitive"]:
                harmful_score += prob
        else:
            if label.lower() in ["unsafe", "harmful"]:
                harmful_score += prob
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