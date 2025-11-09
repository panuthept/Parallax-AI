from typing import List
from .metrics import SafetyMetrics
from ...modules.safeguard_modules import BaseGuardModule


class SafetyBenchmark:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir

    def _get_samples(self) -> List[dict]:
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self, safeguard: BaseGuardModule, **kwargs) -> List[dict]:
        samples: List[dict] = self._get_samples(**kwargs)
        return safeguard.run(inputs=samples, verbose=True)
    
    def evaluate(self, safeguard, threshold: float = 0.5, **kwargs) -> SafetyMetrics:
        samples = self.run(safeguard, **kwargs)
        metrics = SafetyMetrics(samples=samples, threshold=threshold)
        return {
            "performance": metrics.get_results(),
            "samples": [sample for sample in samples],
        }