import lancedb
import pandas as pd
from tabulate import tabulate
from rag_helper.db import get_table
from rag_helper.data import get_labels
from rag_helper.models import QueryItem, FormatType
from rag_helper.query import fts_search, vector_search
from rag_helper.eval import score_retrieval, calculate_precision, calculate_recall, calculate_reciprocal_rank
from rag_helper.embeddings import EmbeddingModel

sizes = [3]
eval_fns = {"mrr": calculate_reciprocal_rank, "recall": calculate_recall, "precision": calculate_precision}

db = lancedb.connect("../lance")
table = get_table(db, "ms_marco")

queries = get_labels('./data/queries_single_label.jsonl')[:30]
queries = [
    QueryItem(
        **{"query": item["query"], "selected_chunk_ids": [item["selected_chunk_ids"]]}
    )
    for item in queries
]

candidates = {"Full Text Search": fts_search, "Semantic Search": vector_search}

results = {}

for candidate, search_fn in candidates.items():
    search_results = search_fn(table, queries, 10)
    chunk_ids = [[item["selected_chunk_ids"] for item in result] for result in search_results]
    print(queries[0])
    lables = queries[0].selected_chunk_ids
    print(chunk_ids[0])
    evaluation_metrics = score_retrieval(chunk_ids[0], lables, sizes, eval_fns)
    print(evaluation_metrics)
    evaluation_metrics = [
        score_retrieval(retrieved_chunk_ids, query.selected_chunk_ids, sizes, eval_fns)
        for retrieved_chunk_ids, query in zip(chunk_ids, queries)
    ]
    results[f"{candidate}"] = pd.DataFrame(evaluation_metrics).mean()

# Convert the dictionary to a DataFrame
df = pd.DataFrame(results)

# Print the table
print(tabulate(df.round(2), headers="keys", tablefmt="grid"))
# model = EmbeddingModel(
#     model_name_or_path="data/onnx",
#     format=FormatType.onnx,
#     file_name='model_quantized.onnx'
# )

# test_semantic = table.search(
#     model.encode("what's a good place to visit in France?"),
#     query_type="vector"
# ).limit(10).select(['query']).to_list()
# for result in test_semantic:
#     print(
#         f"(Semantic search) text: {result}"
#     )


# test_full = table.search(
#     "what's a good place to visit in France?"
# ).limit(10).select(['query']).to_list()
# for result in test_full:
#     print(
#         f"(Full Text Search) text: {result}"
#     )