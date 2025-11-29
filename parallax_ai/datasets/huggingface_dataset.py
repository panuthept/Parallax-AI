from .base_class import Dataset
from typing import Callable, List


class HuggingFaceDataset(Dataset):
    @classmethod
    def get_samples(
        cls,
        path: str,
        subset: str = None,
        splits: List[str] = None,
        sample_acquisitions: List[Callable[[dict], dict]] = None,
        max_samples: int = None,
        cache_dir: str = None,
    ) -> List[dict]:
        splits = splits or [None]
        sample_acquisitions = sample_acquisitions or [lambda data: data]
        from datasets import load_dataset

        samples = []
        for split in splits:
            dataset = load_dataset(path, subset, split=split, cache_dir=cache_dir)
            for data in dataset:
                for sample_acquisition in sample_acquisitions:
                    # Acquire sample using the provided function
                    sample = sample_acquisition(dict(data))
                    # Skip if sample acquisition returns None
                    if sample is None:
                        continue
                    # Add metadata
                    if "metadata" not in sample:
                        sample["metadata"] = {}
                    sample["metadata"].update({
                            "dataset_name": path, 
                            "subset": subset, 
                            "split": split,
                    })
                    samples.append(sample)
                    # Stop if we have reached max_samples
                    if max_samples is not None and len(samples) >= max_samples:
                        return samples[:max_samples]
        return samples
        