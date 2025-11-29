import numpy as np
from scipy import stats
from typing import List
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc


class SafeguardMetrics:
    def __init__(self, label_mapping: dict, threshold: float = 0.5, **kwargs):
        self.label_mapping = label_mapping
        self.threshold = threshold
    
    def __call__(
        self, 
        predicted_scores: List[float],
        gold_labels: List[str],
        gold_scores: List[float] = None
    ) -> dict:
        gold_labels = [self.label_mapping[gold_label] for gold_label in gold_labels]

        gold_labels = np.array(gold_labels)
        predicted_scores = np.array(predicted_scores)

        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_labels, 
            (predicted_scores >= self.threshold).astype(int), 
            average='binary'
        )

        precision_curve, recall_curve, _ = precision_recall_curve(gold_labels, predicted_scores)
        pr_auc = auc(recall_curve, precision_curve)

        metrics = {
            "supports": len(gold_labels),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "pr_auc": pr_auc
        }

        if gold_scores is not None:
            mean_squared_error = np.mean((gold_scores - predicted_scores) ** 2)
            spearman_corr, _ = stats.spearmanr(gold_scores, predicted_scores)
            metrics["mean_squared_error"] = mean_squared_error.item()
            metrics["spearman_correlation"] = spearman_corr.item()
        return metrics