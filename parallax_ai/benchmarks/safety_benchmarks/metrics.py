import numpy as np
from typing import List
from scipy import stats
from sklearn.metrics import (
    precision_recall_fscore_support, 
    precision_recall_curve, 
    auc
)


class SafetyMetrics:
    def __init__(self, samples: List[dict], threshold: float = 0.5):
        self.samples = samples
        self.threshold = threshold

    def get_results(self):
        y_true = []
        y_scores = []
        y_true_binary = []

        for sample in self.samples:
            if sample["gold_harmful_label"] is not None and sample["gold_severity_level"] is not None and sample["harmful_score"] is not None:
                y_true.append(sample["gold_severity_level"])
                y_scores.append(sample["harmful_score"])
                y_true_binary.append(sample["gold_harmful_label"])

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        # y_true_binary = (y_true >= self.threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, 
            (y_scores >= self.threshold).astype(int), 
            average='binary'
        )

        precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_scores)
        pr_auc = auc(recall_curve, precision_curve)

        mean_squared_error = np.mean((y_true - y_scores) ** 2)
        spearman_corr, _ = stats.spearmanr(y_true, y_scores)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "pr_auc": pr_auc,
            "mean_squared_error": mean_squared_error.item(),
            "spearman_correlation:": spearman_corr.item(),
            "supports": len(y_true),
        }