from lancedb.table import Table
from tqdm import tqdm
from lancedb.rerankers import LinearCombinationReranker, CrossEncoderReranker
import instructor
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import Literal
from tqdm.asyncio import tqdm_asyncio as asyncio
from .string_helpers import strip_punctuation
from .models import QueryItem
from .gemini_helper import GeminiHelper
from .embeddings import EmbeddingModel 
from .models import FormatType

def fts_search(table: Table, queries: list[QueryItem], top_k: int):
    data = []
    for query in tqdm(queries, desc="Excuting Full Text search now..."):
        items = (
            table.search(strip_punctuation(query.query), query_type="fts")
            .limit(top_k)
            .select(["query", "selected_chunk_ids"])
            .to_list()
        )
        data.append(items)
    return data

def vector_search(
    table: Table, queries: list[QueryItem], top_k: int, batch_size: int = 20, 
):  
    model = EmbeddingModel("data/onnx", 
                            format=FormatType.onnx, 
                            file_name='model_quantized.onnx')
    embedded_queries = model.generate_embeddings(queries, batch_size=batch_size)
    return [
        table.search(query_embedding, query_type="vector")
        .limit(top_k)
        .select(["selected_chunk_ids"])
        .to_list()
        for query_embedding in tqdm(embedded_queries, desc="Executing Vector search now...")
    ]

def hybird_search(
    table: Table, queries: list[QueryItem], top_k: int, batch_size: int = 20,
):
    return [
        table.search(strip_punctuation(query.query), query_type='hybrid')
        .limit(top_k)
        .to_list()
        for query in tqdm(queries, desc="Executing Hybrid search now...")
    ]

def linear_combination_search(
    table: Table, queries: list[QueryItem], top_k: int, vector_search_weight: float
):
    reranker = LinearCombinationReranker(vector_search_weight)
    return [
        table.search(strip_punctuation(query.query), query_type='hybrid')
        .rerank(reranker)
        .limit(top_k)
        .to_list()
        for query in tqdm(queries, desc=f"Linear Combination (weight {vector_search_weight})")
    ]

def Cross_rerank_search(
    table: Table, queries: list[QueryItem], top_k: int, model_name: str, query_type='fts',
):
    cross_reranker = CrossEncoderReranker(model_name)
    return [
        table.search(strip_punctuation(query.query), query_type=query_type)
        .rerank(cross_reranker)
        .limit(top_k)
        .to_list()
        for query in tqdm(queries, desc=f"Cross Rerank search with {model_name}")
    ]

async def classify_queries(queries: list[str], GOOGLE: str, client):
    client = client
    category_description = """
    This represents a categorization of the user's query

    - 'support': Covers queries that are asking for support or help
    - 'information': Covers queries that are asking for information
    """
    class Category(BaseModel):
        category: Literal['support', 'information'] = Field(
            ...,
            description=category_description,
        )
    async def classify(query: str):
        return await client.messages.create(
            messages= [
                {
                    "role": "system",
                    "content": "You are expert topic classifier. Your job is to classify the following title into the categories provided in the response object. Make sure to classify it into one of the individual categories provided",
                },
                {
                    "role": "user",
                    "content": f"The title is {query}",
                },
            ],
            response_model=Category,
        )
    coros = [classify(query) for query in queries]
    result = await asyncio.gather(*coros)
    return result