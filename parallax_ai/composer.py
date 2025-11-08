from typing import Callable
from dataclasses import dataclass


@dataclass
class OutputComposer:
    name: str
    dependencies: list[str]
    condition: Callable[[dict], bool]
    compose: Callable[[dict], dict]
    
    def condition_satisfy(self, instance_contents: dict) -> bool:
        if self.condition is None:
            return True
        return self.condition(instance_contents)