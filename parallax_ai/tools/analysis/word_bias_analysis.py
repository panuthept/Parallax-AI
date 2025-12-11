import math
import json
from typing import List, Dict
from collections import defaultdict


class WordBiasAnalysis:
    def __init__(self, load_path: str = None):
        self.keyword_stats = None
        if load_path:
            self.load(load_path)

    def get_available_labels(self) -> List[str]:
        if self.keyword_stats is None:
            return []
        return list(self.keyword_stats.keys())
    
    def get_keyword_stats(self, label: str) -> Dict[str, Dict[str, float]]:
        assert self.keyword_stats is not None, "Keyword statistics have not been computed or loaded."
        assert label in self.keyword_stats, f"Unknown label '{label}', please choose from {list(self.keyword_stats.keys())}."
        return self.keyword_stats[label]

    def load(self, path: str):
        assert path.endswith(".json"), "Only JSON format is supported for loading."
        with open(path, "r") as f:
            self.keyword_stats = json.load(f)

    def save(self, path: str):
        assert path.endswith(".json"), "Only JSON format is supported for saving."
        with open(path, "w") as f:
            json.dump(self.keyword_stats, f, indent=2)

    def fit(self, samples: List[dict]):
        # Frequency tables
        word_freq = defaultdict(int)                     # count(W)
        label_freq = defaultdict(int)                    # count(Y)
        label_word_freq = defaultdict(lambda: defaultdict(int))  # count(W,Y)

        # Collect frequencies
        for sample in samples:
            label = sample["label"]
            words = sample["text"]

            label_freq[label] += 1
            for w in words:
                word_freq[w] += 1
                label_word_freq[label][w] += 1

        # Global corpus statistics
        total_words = sum(word_freq.values())            # total word tokens
        total_labels = sum(label_freq.values())          # number of samples (documents)

        results = {}
        # Compute statistics per label
        for label in label_freq:
            p_Y = label_freq[label] / total_labels       # P(Y)
            total_words_Y = sum(label_word_freq[label].values())

            label_results = {}

            for w, count_WY in label_word_freq[label].items():
                # Basic probabilities
                p_W = word_freq[w] / total_words                      # P(W)
                p_W_given_Y = count_WY / total_words_Y               # P(W|Y)
                p_WY = count_WY / total_words                        # P(W,Y)

                # PMI and LMI
                PMI = math.log(p_WY / (p_W * p_Y))
                LMI = count_WY * PMI                                 # freq × PMI

                label_results[w] = {
                    "PMI": PMI,
                    "LMI": LMI,
                }
            results[label] = label_results
        self.keyword_stats = results

    def predict(self, sample: dict) -> Dict[str, float]:
        assert self.keyword_stats is not None, "Keyword statistics have not been computed or loaded."
        text = sample["text"]

        class_scores = {label: 0.0 for label in self.get_available_labels()}
        for label in class_scores:
            stats = self.get_keyword_stats(label)
            for w in text:
                class_scores[label] += stats.get(w, {"LMI": 0.0})["LMI"]
        class_scores = {label: math.exp(score) for label, score in class_scores.items()}
        class_probs = {label: score / sum(class_scores.values()) for label, score in class_scores.items()}
        return class_probs


if __name__ == "__main__":
    samples = [
        {"text": ["Is", "41", "a", "prime", "number", "?"], "label": "Safe"},
        {"text": ["How", "to", "make", "a", "bomb", "?"], "label": "Harmful"},
    ]
    tool = WordBiasAnalysis()
    tool.fit(samples)
    for label in tool.get_available_labels():
        print(f"Label: {label}")
        for keyword, stats in tool.get_keyword_stats(label).items():
            print(f"  Keyword: {keyword}, PMI: {stats['PMI']:.4f}, LMI: {stats['LMI']:.4f}")
    print()