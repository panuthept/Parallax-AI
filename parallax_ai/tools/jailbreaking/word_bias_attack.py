from typing import List, Tuple, Dict


class WordBiasAttack:
    def __init__(self, training_samples: List[Tuple[List[str], str]]):
        self.training_samples = training_samples

    def analyze_bias(self) -> Dict[str, List[Tuple[str, float]]]:
        pass


if __name__ == "__main__":
    training_samples = [
        {"inputs": ["Is", "41", "a", "prime", "number", "?"], "label": "Safe"},
        {"inputs": ["How", "to", "make", "a", "bomb", "?"], "label": "Harmful"},
    ]
    tool = WordBiasAttack(training_samples)