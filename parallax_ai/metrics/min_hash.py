import numpy as np
from typing import List
from random import randint
from scipy.spatial.distance import cosine


class MinHash:
    def __init__(self, num_hashes: int = 128, max_value: int = 2**32 - 1, prime: int = 4294967311):
        self.num_hashes = num_hashes
        self.max_value = max_value
        self.prime = prime

    def hash(self, input_set: set) -> List[int]:
        hash_values = [self.max_value] * self.num_hashes
        a_coeffs = [randint(1, self.prime - 1) for _ in range(self.num_hashes)]
        b_coeffs = [randint(0, self.prime - 1) for _ in range(self.num_hashes)]

        for item in input_set:
            item_hash = hash(item)
            for i in range(self.num_hashes):
                combined_hash = (a_coeffs[i] * item_hash + b_coeffs[i]) % self.prime
                if combined_hash < hash_values[i]:
                    hash_values[i] = combined_hash
        return hash_values
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        hash1 = self.hash(set1)
        hash2 = self.hash(set2)
        identical_hashes = sum(1 for h1, h2 in zip(hash1, hash2) if h1 == h2)
        return identical_hashes / self.num_hashes
    
    def cosine_similarity(self, set1: set, set2: set) -> float:
        hash1 = np.array(self.hash(set1))
        hash2 = np.array(self.hash(set2))
        return 1 - cosine(hash1, hash2)