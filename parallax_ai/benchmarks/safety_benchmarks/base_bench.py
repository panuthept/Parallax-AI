from typing import List, Union
from ...service import Service
from .metrics import SafetyMetrics
from ...modules.safeguards import BaseGuardModule


class SafetyBenchmark:
    default_label_mapping = {
        "Harmful": 1,
        "Unsafe": 1,
        "Safe": 0
    }
    
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir

    def _get_samples(self, **kwargs) -> List[dict]:
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self, safeguard: Union[Service, BaseGuardModule], debug_mode: bool = False, verbose: bool = True, **kwargs) -> List[dict]:
        samples: List[dict] = self._get_samples(**kwargs)
        return safeguard.run(inputs=samples, debug_mode=debug_mode, verbose=verbose)
    
    def evaluate(self, safeguard: Union[Service, BaseGuardModule], label_mapping: dict = None, threshold: float = 0.5, **kwargs) -> dict:
        label_mapping = self.default_label_mapping if label_mapping is None else label_mapping
        
        samples = self.run(safeguard, **kwargs)
        metrics = SafetyMetrics(samples=samples, label_mapping=label_mapping, threshold=threshold)
        return {
            "performance": metrics.get_results(),
            "examples": [sample for sample in samples],
        }