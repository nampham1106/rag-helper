import lancedb
import logging
from rag_helper.query import fts_search
from rag_helper.models import QueryItem, MacroModel
from rag_helper.eval import score_retrieval, calculate_precision, calculate_recall, calculate_reciprocal_rank
from rag_helper.db import get_table, insert_data_into_table
from rag_helper.data import get_labels
from rag_helper.embeddings import EmbeddingModel
from rag_helper.models import FormatType

logger = logging.getLogger(__name__)
model = EmbeddingModel(
    model_name_or_path="data/onnx",
    format=FormatType.onnx,
    file_name='model_quantized.onnx'
)
db = lancedb.connect("../lance")
db.drop_table('ms_marco')
queries = get_labels("./data/queries_single_label.jsonl")
table = get_table(db, "ms_marco", MacroModel)

queries = [
    MacroModel(
        **{
            "query": item["query"],
            "selected_chunk_ids": item["selected_chunk_ids"],
            "vector": model.encode(item["query"])}
    )
    for item in queries
]
insert_data_into_table(table, queries, batch_size=20)
table.create_fts_index("query", replace=True)
# print(table)
print(db['ms_marco'].head())
test = queries[0:1]
print("queries: ", test)
result = fts_search(table, test, 3)
print("result search: ")
for res in result:
    for item in res:
        # print(item['query'])
        # print(item['selected_chunk_ids'])
        print(item)