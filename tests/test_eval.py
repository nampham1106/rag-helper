from rag_helper.eval import score_retrieval, calculate_precision, calculate_recall, calculate_reciprocal_rank

if __name__ == "__main__":
    predictions = ["a", "b", "c", "d", "e"]
    label = "b"
    sizes = [3, 5, 10, 15, 25]
    scoring_fns = {
        "calculate_recall": calculate_recall,
        "calculate_reciprocal_rank": calculate_reciprocal_rank,
        "calculate_precision": calculate_precision,
    }
    print(score_retrieval(predictions, label, sizes, scoring_fns))