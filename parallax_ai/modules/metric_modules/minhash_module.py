from typing import List
from random import randint
from dataclasses import dataclass
from ..basic_modules.lambda_module import LambdaModule


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

    def hash(self, inputs: list) -> List[int]:
        hash_values = [float("inf")] * self.num_hashes

        for item in set(inputs):
            item_hash = hash(item) if not isinstance(item, int) else item
            for i in range(self.num_hashes):
                combined_hash = (self.a_coeffs[i] * item_hash + self.b_coeffs[i]) % self.prime
                if combined_hash < hash_values[i]:
                    hash_values[i] = combined_hash
        return hash_values
    
    def jaccard_similarity(self, set1: list, set2: list) -> float:
        hash1 = self.hash(set1)
        hash2 = self.hash(set2)
        identical_hashes = set(hash1).intersection(set(hash2)).__len__()
        return identical_hashes / self.num_hashes
    
    def __call__(self, input_data: dict) -> dict:
        set1 = set(input_data.get("set1", []))
        set2 = set(input_data.get("set2", []))
        similarity = self.jaccard_similarity(set1, set2)
        return {"jaccard_similarity": similarity}
    
@dataclass
class MinHashModule(LambdaModule):
    num_hashes: int = 128
    max_value: int = 2**32 - 1
    prime: int = 4294967311
    seed: int = None

    @classmethod
    def get_input_stucture(cls) -> str:
        return "{'set1': list, 'set2': list}"
    
    @classmethod
    def get_output_stucture(cls) -> str:
        return "{'jaccard_similarity': float}"

    def __post_init__(self):
        minhash_instance = MinHash(
            num_hashes=self.num_hashes,
            max_value=self.max_value,
            prime=self.prime,
            seed=self.seed
        )
        self.function = minhash_instance
        super().__post_init__()