from typing import List, Callable, Optional


class Dataset:    
    available_subsets = [None]
    available_splits = [None]

    @classmethod
    def _get_samples(cls, **kwargs) -> List[dict]:
        raise NotImplementedError("Subclasses must implement _get_samples method.")

    @classmethod
    def get_samples(cls, transformation: Optional[Callable] = None, **kwargs) -> List[dict]:
        samples = cls._get_samples(**kwargs)
        return transformation(samples) if transformation else samples