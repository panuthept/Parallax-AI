from typing import List
from ..base_class import Dataset
from ..huggingface_dataset import HuggingFaceDataset


class SEALSBenchDataset(Dataset):
    available_subsets = [None]
    available_splits = {None: ["train", "validation", "test"]}
    label_mapping = {
        "unsafe": 1.0,
        "safe": 0.0,
    }

    @classmethod
    def get_samples(
        cls,
        splits: List[str] = None,
        **kwargs,
    ) -> List[dict]:
        splits = splits or cls.available_splits[None]
        for split in splits:
            assert split in cls.available_splits[None], f"Split {split} not available. Choose from {cls.available_splits[None]}."

        return HuggingFaceDataset.get_samples(
            path="MickyMike/SEALSBench",
            splits=splits,
            sample_acquisitions=[lambda data: {
                "prompt": data["prompt"],
                "gold_harmful_label": int(cls.label_mapping[data["label"]]),
                "gold_severity_level": float(cls.label_mapping[data["label"]]),
                "metadata": {
                    "task": "prompt_classification",
                    "category": data["category"],
                    "origin": data["origin"],
                    "target_language": data["target_language"]
                },
            }],
            **kwargs,
        )