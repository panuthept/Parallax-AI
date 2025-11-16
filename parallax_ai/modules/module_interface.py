from types import GeneratorType
from dataclasses import dataclass
from ..dataclasses import Instance
from typing import Optional, List, Callable, Any


@dataclass
class ModuleInterface:
    dependencies: List[str]
    input_processing: Optional[Callable[[dict], list]] = None # (deps) -> inputs
    output_processing: Callable[[list, list], list] = None # (inputs, outputs) -> deps

    def get_module_outputs(self, inputs: Any, outputs: Any) -> dict:
        if self.output_processing is not None:
            processed_output = self.output_processing(inputs, outputs)
            return processed_output
        return outputs

    def get_module_inputs(self, instance: Instance) -> List[dict]:
        # inputs = {dep: instance.contents[dep] for dep in self.dependencies}
        inputs = instance.contents
        if self.input_processing is not None:
            processed_inputs = self.input_processing(inputs)
            if isinstance(processed_inputs, GeneratorType):
                processed_inputs = list(processed_inputs)
            elif not isinstance(processed_inputs, list):
                processed_inputs = [processed_inputs]
            return processed_inputs
        return [inputs]