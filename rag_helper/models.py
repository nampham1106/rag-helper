from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from enum import Enum

class MacroModel(LanceModel):
    query: str
    selected_chunk_ids: str
    vector: Vector(1024) # type: ignore

class ViHealthModel(LanceModel):
    id: str
    query: str
    answer: str
    link: str
    vector: Vector(1024) # type: ignore

class QueryItem(LanceModel):
    query: str
    selected_chunk_ids: list[str]

class FormatType(Enum):
    sentence_transformers = "sentence-transformers"
    onnx = "onnx"

