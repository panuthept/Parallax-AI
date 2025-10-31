from typing import List
from .metrics import SafetyMetrics
from .dataclasses import SafetySample


class SafetyBenchmark:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir

    def _get_samples(self) -> List[SafetySample]:
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self, safeguard, **kwargs) -> List[SafetySample]:
        samples: List[SafetySample] = self._get_samples(**kwargs)

        inputs = [sample.messages for sample in samples]
        harmful_scores = safeguard(inputs, verbose=True)
        assert len(harmful_scores) == len(samples), "Number of outputs must match number of samples."

        for sample, harmful_score in zip(samples, harmful_scores):
            sample.harmful_score = harmful_score
        return samples
    
    def evaluate(self, safeguard, threshold: float = 0.5, **kwargs) -> SafetyMetrics:
        samples = self.run(safeguard, **kwargs)
        metrics = SafetyMetrics(samples=samples, threshold=threshold)
        return {
            "performance": metrics.get_results(),
            "samples": [sample.__dict__ for sample in samples],
        }