import random
from typing import List


class JaccardSimilarityMetric:
    def __init__(
        self, 
        num_hashes: int = 128, 
        max_value: int = 2**32 - 1, 
        prime: int = 4294967311, 
        seed: int = 42, 
        **kwargs
    ):
        random.seed(seed)

        self.num_hashes = num_hashes
        self.max_value = max_value
        self.prime = prime
        self.a_coeffs = [random.randint(0, self.max_value) for _ in range(self.num_hashes)]
        self.b_coeffs = [random.randint(0, self.max_value) for _ in range(self.num_hashes)]

    def hash(self, string: List[str]) -> List[int]:
        hash_values = [float("inf")] * self.num_hashes

        for item in set(string):
            item_hash = hash(item) if not isinstance(item, int) else item
            for i in range(self.num_hashes):
                combined_hash = (self.a_coeffs[i] * item_hash + self.b_coeffs[i]) % self.prime
                if combined_hash < hash_values[i]:
                    hash_values[i] = combined_hash
        return hash_values
    
    def __call__(self, string1: List[str], string2: List[str]) -> float:
        hash1 = self.hash(string1)
        hash2 = self.hash(string2)
        identical_hashes = set(hash1).intersection(set(hash2)).__len__()
        return identical_hashes / self.num_hashes
    
class TextSimilarityMetrics:
    def __init__(self, *args, **kwargs):
        self.jaccard_similarity_metric = JaccardSimilarityMetric(*args, **kwargs)

    def __call__(self, string1: List[str], string2: List[str]) -> dict:
        return {
            "jaccard_similarity": self.jaccard_similarity_metric(string1, string2)
        }