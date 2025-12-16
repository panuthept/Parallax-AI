import numpy as np
from typing import List
from ..base_class import Dataset
from ..huggingface_dataset import HuggingFaceDataset


class SEASafeguardBenchDataset(Dataset):
    available_subsets = ["general", "cultural_content_generation", "cultural_in_the_wild"]
    available_splits = {
        "general": ["EN", "TA", "TH", "TL", "MS", "IN", "MY", "VI"],
        "cultural_content_generation": ["TA_EN", "TH_EN", "TL_EN", "MS_EN", "IN_EN", "MY_EN", "VI_EN"],
        "cultural_in_the_wild": ["TA_EN", "TH_EN", "TL_EN", "MS_EN", "IN_EN", "MY_EN", "VI_EN"],
    }
    label_mapping = {
        "Harmful": 1.0,
        "Sensitive": 0.5,
        "Safe": 0.0,
    }

    @classmethod
    def _get_samples(
        cls,
        subset: str = None,
        splits: List[str] = None,
        **kwargs,
    ) -> List[dict]:
        if subset is not None:
            assert subset in cls.available_subsets, f"Subset {subset} not available. Choose from {cls.available_subsets}."
            splits = splits or cls.available_splits[subset]
            for split in splits:
                assert split in cls.available_splits[subset], f"Split {split} not available. Choose from {cls.available_splits[subset]}."
            subsets = [subset]
        else:
            subsets = cls.available_subsets

        sample_acquisitions = {
            "general": [
                lambda data: {
                    "prompt": data["prompt"],
                    "gold_harmful_label": data["prompt_label"],
                    "gold_severity_level": cls.label_mapping[data["prompt_label"]],
                    "metadata": {
                        "task": "prompt_classification",
                        "source": data["source"],
                    },
                },
                lambda data: {
                    "prompt": data["prompt"],
                    "response": data["response"],
                    "gold_harmful_label": data["response_label"],
                    "gold_severity_level": cls.label_mapping[data["response_label"]],
                    "metadata": {
                        "task": "response_classification",
                        "source": data["source"],
                    },
                }
            ],
            "cultural_content_generation": [
                lambda data: {
                    "prompt": data["en_prompt"],
                    "gold_harmful_label": data["prompt_label"],
                    "gold_severity_level": np.mean([cls.label_mapping[label] for label in data["prompt_annotations"]]).item(),
                    "metadata": {
                        "task": "prompt_classification",
                        "language": "English",
                    },
                },
                lambda data: {
                    "prompt": data["en_prompt"],
                    "response": data["en_response"],
                    "gold_harmful_label": data["response_label"],
                    "gold_severity_level": np.mean([cls.label_mapping[label] for label in data["response_annotations"]]).item(),
                    "metadata": {
                        "task": "response_classification",
                        "language": "English",
                    },
                },
                lambda data: {
                    "prompt": data["local_prompt"],
                    "gold_harmful_label": data["prompt_label"],
                    "gold_severity_level": np.mean([cls.label_mapping[label] for label in data["prompt_annotations"]]).item(),
                    "metadata": {
                        "task": "prompt_classification",
                        "language": "SEA",
                    },
                },
                lambda data: {
                    "prompt": data["local_prompt"],
                    "response": data["local_response"],
                    "gold_harmful_label": data["response_label"],
                    "gold_severity_level": np.mean([cls.label_mapping[label] for label in data["response_annotations"]]).item(),
                    "metadata": {
                        "task": "response_classification",
                        "language": "SEA",
                    },
                }
            ],
            "cultural_in_the_wild": [
                lambda data: {
                    "prompt": data["en_prompt"],
                    "gold_harmful_label": data["prompt_label"],
                    "gold_severity_level": cls.label_mapping[data["prompt_label"]],
                    "metadata": {
                        "task": "prompt_classification",
                        "topic": data["topic"],
                        "language": "English",
                    },
                },
                lambda data: {
                    "prompt": data["local_prompt"],
                    "gold_harmful_label": data["prompt_label"],
                    "gold_severity_level": cls.label_mapping[data["prompt_label"]],
                    "metadata": {
                        "task": "prompt_classification",
                        "topic": data["topic"],
                        "language": "SEA",
                    },
                },
            ],
        }

        samples = []
        for subset in subsets:
            samples.extend(
                HuggingFaceDataset._get_samples(
                    path="aisingapore/SEASafeguardBench",
                    subset=subset,
                    splits=splits or cls.available_splits[subset],
                    sample_acquisitions=sample_acquisitions[subset],
                    **kwargs,
                )
            )
        return samples