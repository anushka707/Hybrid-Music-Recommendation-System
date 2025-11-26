import numpy as np
from pyspark.ml.evaluation import RegressionEvaluator


def compute_rmse(model, predictions):
    """
    Evaluates ALS regression accuracy.
    """
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="listen_count",
        predictionCol="prediction"
    )
    rmse = evaluator.evaluate(predictions)
    return rmse


def precision_at_k(recommended_items, actual_items, k=10):
    """
    recommended_items: list of item IDs recommended by the model
    actual_items: list of actual items the user interacted with
    """
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(actual_items))
    return hits / k


def recall_at_k(recommended_items, actual_items, k=10):
    recommended_k = recommended_items[:k]
    hits = len(set(recommended_k) & set(actual_items))
    return hits / len(actual_items) if len(actual_items) > 0 else 0


def apk(actual, predicted, k=10):
    """
    Average Precision @ K
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual_list, predicted_list, k=10):
    """
    Mean Average Precision @ K
    """
    return np.mean([
        apk(a, p, k)
        for a, p in zip(actual_list, predicted_list)
    ])


def ndcg_at_k(recommended_items, actual_items, k=10):
    """
    Normalized Discounted Cumulative Gain
    """
    def dcg(scores):
        return np.sum([
            (2**rel - 1) / np.log2(idx + 2)
            for idx, rel in enumerate(scores)
        ])

    scores = [1 if item in actual_items else 0 for item in recommended_items[:k]]

    ideal_scores = sorted(scores, reverse=True)

    return dcg(scores) / (dcg(ideal_scores) + 1e-9)
