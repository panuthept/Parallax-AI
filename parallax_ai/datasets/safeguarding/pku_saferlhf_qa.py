from typing import List
from ..base_class import Dataset
from ..huggingface_dataset import HuggingFaceDataset


class PKUSafeRLHFQADataset(Dataset):
    available_subsets = [None]
    available_splits = {None: ["train", "test"]}

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
            path="PKU-Alignment/PKU-SafeRLHF-QA",
            splits=splits,
            sample_acquisitions=[lambda data: {
                "prompt": data["prompt"],
                "response": data["response"],
                "gold_harmful_label": int(not data["is_safe"]),
                "gold_severity_level": float(data["severity_level"]),
                "metadata": {
                    "task": "response_classification",
                    "prompt_source": data["prompt_source"],
                    "response_source": data["response_source"],
                    "harm_category": [category for category, label in data["harm_category"].items() if label == True],
                },
            }],
            **kwargs,
        )