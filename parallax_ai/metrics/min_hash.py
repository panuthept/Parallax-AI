from typing import List
from random import randint


class MinHash:
    def __init__(self, num_hashes: int = 128, max_value: int = 2**32 - 1, prime: int = 4294967311, seed: int = None):
        # Set random seed for reproducibility
        if seed is not None:
            import random
            random.seed(seed)

        self.num_hashes = num_hashes
        self.max_value = max_value
        self.prime = prime
        self.a_coeffs = [randint(0, self.max_value) for _ in range(self.num_hashes)]
        self.b_coeffs = [randint(0, self.max_value) for _ in range(self.num_hashes)]

    def hash(self, input_set: set) -> List[int]:
        hash_values = [float("inf")] * self.num_hashes

        for item in input_set:
            item_hash = hash(item)
            for i in range(self.num_hashes):
                combined_hash = (self.a_coeffs[i] * item_hash + self.b_coeffs[i]) % self.prime
                if combined_hash < hash_values[i]:
                    hash_values[i] = combined_hash
        return hash_values
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        hash1 = self.hash(set1)
        hash2 = self.hash(set2)
        identical_hashes = set(hash1).intersection(set(hash2)).__len__()
        return identical_hashes / self.num_hashes