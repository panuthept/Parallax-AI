import numpy as np
from ..base_module import Job
from dataclasses import dataclass
from collections import defaultdict
from ...utilities import get_dummy_output
from .agent_module import AgentModule, AgentSpec, agent_completions


def agentic_classification(inputs: dict) -> dict:
    predicted_classes = defaultdict(lambda: defaultdict(int))
    for parsed_output in agent_completions(inputs, n=inputs["n"], return_logprobs=False):
        if isinstance(parsed_output, dict):
            for key, value in parsed_output.items():
                predicted_classes[key][value] += 1
    predicted_classes = {key: dict(predicted_classes[key]) for key in predicted_classes}

    softmax_outputs = {}
    for key, class_counts in predicted_classes.items():
        total_counts = sum(class_counts.values())
        class_probabilities = {cls: count / total_counts for cls, count in class_counts.items()}
        softmax_outputs[key] = {k: v for k, v in sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)}
    return softmax_outputs

@dataclass
class AgenticClassificationModule(AgentModule):
    n: int = 10

    def get_executor_input(self, module_input: dict) -> dict:
        executor_input = super().get_executor_input(module_input)
        executor_input["n"] = self.n
        return executor_input
    
    def _create_job(self, instance_id: str, module_input: dict) -> Job:
        return Job(
            module_input=module_input,
            executor_func=agentic_classification,
            executor_input=self.get_executor_input(module_input),
            executor_default_output=get_dummy_output(self.spec.output_structure) if self.spec.default_output is None else self.spec.default_output,
            instance_id=instance_id,
            module_name=self.name,
            progress_name=self.progress_name
        )
    
    
def classification(inputs: dict) -> dict:
    _, tokens_logprobs = agent_completions(inputs, return_logprobs=True)

    classes_logprob = defaultdict(float)
    for label, representative_tokens in inputs["representative_tokens"].items():
        class_logprobs = []
        for i, representative_token in enumerate(representative_tokens):
            if i >= len(tokens_logprobs):
                break
            class_logprob = None
            for token, logprob in tokens_logprobs[i]:
                if token == representative_token:
                    class_logprob = logprob
                    break  # Only consider the first occurrence of any of the
            if class_logprob is None:
                break
            class_logprobs.append(class_logprob)
        if len(class_logprobs) > 0:
            classes_logprob[label] = np.sum(class_logprobs).item()

    if len(classes_logprob) == 0:
        raise ValueError("Agent classification failed to produce any valid outputs.")
    classes_logprob = dict(classes_logprob)
        
    total = sum(np.exp(list(classes_logprob.values())))
    softmax_outputs = {label: 0.0 for label in inputs["representative_tokens"].keys()}
    for label, logprob in classes_logprob.items():
        softmax_outputs[label] = (np.exp(logprob) / total).item()
    softmax_outputs = {k: v for k, v in sorted(softmax_outputs.items(), key=lambda item: item[1], reverse=True)}
    return softmax_outputs
    
@dataclass
class ClassificationSpec(AgentSpec):
    representative_tokens: dict = None

@dataclass
class ClassificationModule(AgentModule):
    spec: ClassificationSpec = None

    def get_executor_input(self, module_input: dict) -> dict:
        executor_input = super().get_executor_input(module_input)
        executor_input["representative_tokens"] = self.spec.representative_tokens
        return executor_input
    
    def _create_job(self, instance_id: str, module_input: dict) -> Job:
        return Job(
            module_input=module_input,
            executor_func=classification,
            executor_input=self.get_executor_input(module_input),
            executor_default_output=get_dummy_output(self.spec.output_structure) if self.spec.default_output is None else self.spec.default_output,
            instance_id=instance_id,
            module_name=self.name,
            progress_name=self.progress_name
        )