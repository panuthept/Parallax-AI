import numpy as np
from typing import Optional
from ..dataclasses import SafetySample
from ..baseclasses import SafetyBenchmark


class SEASafeguardBench(SafetyBenchmark):
    available_subsets_splits = {
        "general": ["EN", "TA", "TH", "TL", "MS", "IN", "MY", "VI"],
        "cultural_content_generation": ["TA_EN", "TH_EN", "TL_EN", "MS_EN", "IN_EN", "MY_EN", "VI_EN"],
        "cultural_in_the_wild": ["TA_EN", "TH_EN", "TL_EN", "MS_EN", "IN_EN", "MY_EN", "VI_EN"],
    }
    harmful_score_mapping = {
        "Harmful": 1.0,
        "Sensitive": 0.5,
        "Safe": 0.0,
    }

    def _get_samples(self, subset: str, split: str, language: Optional[str] = None, task: Optional[str] = None) -> list:
        assert subset in self.available_subsets_splits, f"Invalid subset: {subset}. Valid subsets are {self.available_subsets_splits.keys()}"
        assert split in self.available_subsets_splits[subset], f"Invalid split: {split}. Valid splits are {self.available_subsets_splits[subset]}"
        assert language in [None, "English", "Local"], f"Invalid language: {language}. Valid languages are [None, 'English', 'Local']"
        assert task in [None, "prompt_classification", "response_classification"], f"Invalid task: {task}. Valid tasks are [None, 'prompt_classification', 'response_classification']"
        languages = ["English", "Local"] if language is None else [language]
        tasks = ["prompt_classification", "response_classification"] if task is None else [task]

        from datasets import load_dataset
        dataset = load_dataset("aisingapore/SEASafeguardBench", subset, split=split, cache_dir=self.cache_dir)

        samples = []
        for data in dataset:
            if subset == "general":
                if "prompt_classification" in tasks:
                    samples.append(
                        SafetySample(
                            messages=[{"role": "user", "content": data["prompt"]}],
                            gold_harmful_score=self.harmful_score_mapping[data["prompt_label"]],
                        )
                    )
                if "response_classification" in tasks and data["response"] is not None:
                    samples.append(
                        SafetySample(
                            messages=[
                                {"role": "user", "content": data["prompt"]}, 
                                {"role": "assistant", "content": data["response"]}
                            ],
                            gold_harmful_score=self.harmful_score_mapping[data["response_label"]],
                        )
                    )
            elif subset == "cultural_content_generation":
                if "prompt_classification" in tasks:
                    if "English" in languages:
                        samples.append(
                            SafetySample(
                                messages=[{"role": "user", "content": data["en_prompt"]}],
                                gold_harmful_score=np.mean([self.harmful_score_mapping[label] for label in data["prompt_annotations"]]).item(),
                            )
                        )
                    if "Local" in languages:
                        samples.append(
                            SafetySample(
                                messages=[{"role": "user", "content": data["local_prompt"]}],
                                gold_harmful_score=np.mean([self.harmful_score_mapping[label] for label in data["prompt_annotations"]]).item(),
                            )
                        )
                if "response_classification" in tasks and data["en_response"] is not None and data["local_response"] is not None:
                    if "English" in languages:
                        samples.append(
                            SafetySample(
                                messages=[
                                    {"role": "user", "content": data["en_prompt"]},
                                    {"role": "assistant", "content": data["en_response"]}
                                ],
                                gold_harmful_score=np.mean([self.harmful_score_mapping[label] for label in data["response_annotations"]]).item(),
                            )
                        )
                    if "Local" in languages:
                        samples.append(
                            SafetySample(
                                messages=[
                                    {"role": "user", "content": data["local_prompt"]},
                                    {"role": "assistant", "content": data["local_response"]}
                                ],
                                gold_harmful_score=np.mean([self.harmful_score_mapping[label] for label in data["response_annotations"]]).item(),
                            )
                        )
            elif subset == "cultural_in_the_wild":
                if "English" in languages:
                    samples.append(
                        SafetySample(
                            messages=[{"role": "user", "content": data["en_prompt"]}],
                            gold_harmful_score=self.harmful_score_mapping[data["prompt_label"]],
                        )
                    )
                if "Local" in languages:
                    samples.append(
                        SafetySample(
                            messages=[{"role": "user", "content": data["local_prompt"]}],
                            gold_harmful_score=self.harmful_score_mapping[data["prompt_label"]],
                        )
                    )
        return samples