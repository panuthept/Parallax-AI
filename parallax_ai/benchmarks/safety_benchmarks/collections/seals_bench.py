from typing import Optional
from ..base_bench import SafetyBenchmark


class SEALSBench(SafetyBenchmark):
    harmful_label_mapping = {
        "unsafe": 1,
        "safe": 0,
    }
    available_language = ["English", "Lao", "Chinese", "Vietnamese", "Indonesian", "Thai", "Malay", "Khmer", "Burmese", "Filipino"]

    def _get_samples(self, language: Optional[str] = None, **kwargs) -> list:
        assert language in [None, *self.available_language], f"Invalid language: {language}. Valid languages are [None, 'English', 'Lao', 'Chinese', 'Vietnamese', 'Indonesian', 'Thai', 'Malay', 'Khmer', 'Burmese', 'Filipino']"

        from datasets import load_dataset
        dataset = load_dataset("MickyMike/SEALSBench", split="test", cache_dir=self.cache_dir)

        samples = []
        for data in dataset:
            if language is not None and data["target_language"] != language:
                continue
            samples.append({
                "prompt": data["prompt"], 
                "gold_harmful_label": int(self.harmful_label_mapping[data["label"]]),
                "gold_severity_level": float(self.harmful_label_mapping[data["label"]]),
            })
        return samples