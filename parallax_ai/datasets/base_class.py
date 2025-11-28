from typing import List


class Dataset:    
    def __init__(self, cache_dir: str = None, max_samples: int = None):
        self.cache_dir = cache_dir
        self.max_samples = max_samples

    def _get_samples(self, *args, **kwargs) -> List[dict]:
        raise NotImplementedError("Subclasses should implement this method.")

    def get_samples(self, *args, **kwargs) -> List[dict]:
        samples = self._get_samples(*args, **kwargs)
        if self.max_samples is not None:
            return samples[:self.max_samples]
        return samples