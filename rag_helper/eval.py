from itertools import product
from typing import Callable

SIZES = [3, 5, 10, 15, 25]

def accuracy(predictions, labels):
    correct_predictions = sum(1 for prediction in predictions if prediction in labels)
    return correct_predictions / len(labels)

def calculate_recall(predictions, labels):
    correct_predictions = sum(1 for label in labels if label in predictions)
    if labels:
        return correct_predictions / len(labels)
    return 0

def calculate_reciprocal_rank(predictions, labels):
    for index, prediction in enumerate(predictions):
        if prediction in labels:
            return 1 / (index + 1)
    return 0

def calculate_precision(predictions, labels):
    correct_predictions = sum(1 for prediction in predictions if prediction in labels)
    if predictions:
        return correct_predictions / len(predictions)
    return 0

def score_retrieval(
    preds: list[str],
    label: str | list[str],
    sizes: list[int],
    scoring_fns: dict[str, Callable[[list[str], list[str]], float]],
):
    return {
        f"{fn_name}@{size}": round(
            scoring_fns[fn_name](
                preds[:size], [label] if isinstance(label, str) else label
            ),
            3,
        )
        for fn_name, size in product(scoring_fns.keys(), sizes)
    }