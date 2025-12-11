import math
from typing import List
from collections import defaultdict


class WordBiasAnalysis:
    def analyze(self, samples: List[dict]):
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
        return results


if __name__ == "__main__":
    samples = [
        {"text": ["Is", "41", "a", "prime", "number", "?"], "label": "Safe"},
        {"text": ["How", "to", "make", "a", "bomb", "?"], "label": "Harmful"},
    ]
    tool = WordBiasAnalysis()
    results = tool.analyze(samples)
    for label in results:
        print(f"Label: {label}")
        for word, stats in results[label].items():
            print(f"  Word: {word}, PMI: {stats['PMI']:.4f}, LMI: {stats['LMI']:.4f}")
    print()