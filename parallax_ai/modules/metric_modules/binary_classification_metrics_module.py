import numpy as np
from scipy import stats
from dataclasses import dataclass
from ..basic_modules.lambda_module import LambdaModule
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc


class BinaryClassificationMetrics:
    def __init__(self, label_mapping: dict, threshold: float = 0.5):
        self.label_mapping = label_mapping
        self.threshold = threshold
    
    def __call__(self, input_data: dict) -> dict:
        assert "gold_labels" in input_data and "predicted_scores" in input_data, "Input data must contain 'gold_labels' and 'predicted_scores'."
        gold_labels = input_data["gold_labels"]
        gold_scores = input_data.get("gold_scores", [])
        predicted_scores = input_data["predicted_scores"]

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

        if "gold_scores" in input_data:
            gold_scores = input_data["gold_scores"]
            mean_squared_error = np.mean((gold_scores - predicted_scores) ** 2)
            spearman_corr, _ = stats.spearmanr(gold_scores, predicted_scores)
            metrics["mean_squared_error"] = mean_squared_error.item()
            metrics["spearman_correlation"] = spearman_corr.item()
        return metrics
    
@dataclass
class BinaryClassificationMetricsModule(LambdaModule):
    label_mapping: dict
    threshold: float = 0.5

    @classmethod
    def get_input_stucture(cls) -> str:
        return "{'set1': list, 'set2': list}"
    
    @classmethod
    def get_output_stucture(cls) -> str:
        return "{'jaccard_similarity': float}"

    def __post_init__(self):
        self.function = BinaryClassificationMetrics(
            label_mapping=self.label_mapping,
            threshold=self.threshold
        )
        super().__post_init__()