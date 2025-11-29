from typing import List


class Dataset:    
    available_subsets = [None]
    available_splits = [None]

    @classmethod
    def get_samples(cls) -> List[dict]:
        raise NotImplementedError("Subclasses must implement get_samples method.")