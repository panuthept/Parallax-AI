from dataclasses import dataclass
from collections import defaultdict
from ..utilities import get_dummy_output
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor as Pool
from .agent_module import AgentJob, AgentModule, agent_completions


def agent_classification(inputs: dict) -> dict:
    pool = Pool(max_workers=inputs["n"])
    running_tasks = pool.submit(agent_completions, [inputs] * inputs["n"])
    
    predicted_classes = defaultdict(lambda: defaultdict(int))
    for future in as_completed(running_tasks):
        parsed_output = future.result()
        if isinstance(parsed_output, dict):
            for key, value in parsed_output.items():
                predicted_classes[key][value] += 1
        else:
            raise ValueError("Parsed output must be a dict for classification.")

    softmax_outputs = {}
    for key, class_counts in predicted_classes.items():
        total_counts = sum(class_counts.values())
        class_probabilities = {cls: count / total_counts for cls, count in class_counts.items()}
        softmax_outputs[key] = sorted(class_probabilities.items(), key=lambda item: item[1], reverse=True)
    return softmax_outputs

@dataclass
class ClassificationModule(AgentModule):
    n: int = 10

    def get_executor_input(self, module_input: dict) -> dict:
        executor_input = super().get_executor_input(module_input)
        executor_input["n"] = self.n
        return executor_input
    
    def _create_job(self, instance_id: str, module_input: dict) -> AgentJob:
        return AgentJob(
            module_input=module_input,
            executor_func=agent_classification,
            executor_input=self.get_executor_input(module_input),
            executor_default_output=get_dummy_output(self.spec.output_structure),
            instance_id=instance_id,
            module_name=self.name,
            progress_name=self.progress_name
        )