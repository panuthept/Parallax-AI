from typing import List
from ..base_bench import SafetyBenchmark


class RabakBench(SafetyBenchmark):
    def _get_samples(self) -> List[dict]:

        from datasets import load_dataset
        dataset = load_dataset("MickyMike/SEALSBench", split="train", cache_dir=self.cache_dir)

        samples = []
        for data in dataset:
            samples.append({"prompt": data["text"], "gold_harmful_score": float(data["binary"])})
        return samples