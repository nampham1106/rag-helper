import json
from rag_helper.data import get_labels
from rag_helper.embeddings import EmbeddingModel
from rag_helper.models import QueryItem, FormatType
from transformers import AutoTokenizer
if __name__ == "__main__":
    model = EmbeddingModel(
        model_name_or_path="data/onnx",
        format=FormatType.onnx,
        file_name='model_quantized.onnx'
    )
    queries = get_labels('./data/queries_single_label.jsonl')[:10]
    print(queries)
    queries = [
        QueryItem(
            **{"query": item["query"], "selected_chunk_ids": [item["selected_chunk_ids"]]}
        )
        for item in queries
    ]
    embeddings = model.generate_embeddings(queries, batch_size=2)
    print(len(embeddings), len(embeddings[0]))