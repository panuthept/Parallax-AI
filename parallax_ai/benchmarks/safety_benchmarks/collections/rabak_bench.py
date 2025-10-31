from ..dataclasses import SafetySample
from ..baseclasses import SafetyBenchmark


class RabakBench(SafetyBenchmark):
    def _get_samples(self) -> list:

        from datasets import load_dataset
        dataset = load_dataset("MickyMike/SEALSBench", split="train", cache_dir=self.cache_dir)

        samples = []
        for data in dataset:
            samples.append(
                SafetySample(
                    messages=[{"role": "user", "content": data["text"]}],
                    gold_harmful_score=float(data["binary"]),
                )
            )
        return samples