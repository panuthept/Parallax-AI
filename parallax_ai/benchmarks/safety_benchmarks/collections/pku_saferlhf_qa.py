from ..base_bench import SafetyBenchmark


class PKUSafeRLHFQA(SafetyBenchmark):
    def _get_samples(self, **kwargs) -> list:
        from datasets import load_dataset
        dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF-QA", split="test", cache_dir=self.cache_dir)

        samples = []
        for data in dataset:
            samples.append({
                "prompt": data["prompt"], 
                "response": data["response"], 
                "gold_harmful_label": int(not data["is_safe"]),
                "gold_severity_level": float(data["severity_level"]),
            })
        return samples