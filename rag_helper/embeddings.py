import os
import logging
import torch
from itertools import batched
from tqdm import tqdm
from typing import Optional
from sentence_transformers import SentenceTransformer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch.nn.functional as F
from .models import QueryItem, FormatType


logger = logging.getLogger(__name__)
class EmbeddingModel(object):
    def __init__(self, 
        model_name_or_path: str,
        format: FormatType,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
    ):
        print(format)
        if format == None:
                raise ValueError("Format is required")
        else:
            logger.info("Format: {}".format(format))
            self.format_model = format

        if model_name_or_path is not None and model_name_or_path != "": 
            logger.info("Load pretrained {}: {}".format(format, model_name_or_path))
            if os.path.exists(model_name_or_path):
                model_path = model_name_or_path
            else:
                raise ValueError("Model path does not exist: {}".format(model_name_or_path))

            if format == FormatType.onnx:
                self.model = ORTModelForFeatureExtraction.from_pretrained(model_path, file_name=file_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            else: 
                self.model = SentenceTransformer(model_path)

    def _mean_pooling(self, model_output, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode(self, query: str):
        if self.format_model == FormatType.sentence_transformers:
            return self.model.encode([query])
        else:
            inputs = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                model_outputs = self.model(**inputs)
            embeddings = self._mean_pooling(model_outputs['last_hidden_state'], inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.tolist()[0]
        
    def generate_embeddings(self, data: list[QueryItem], batch_size: int):
        batches = batched(data, batch_size)

        def generate_embeddings_batch(batch):
            if self.format_model == FormatType.sentence_transformers:
                embeddings = self.model.encode(
                    [item.query for item in list(batch)]
                )
                return embeddings.tolist()
            else:
                inputs = self.tokenizer(
                    [item.query for item in list(batch)],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    model_outputs = self.model(**inputs)
                embeddings = self._mean_pooling(model_outputs['last_hidden_state'], inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                return embeddings.tolist()
            
        batched_embeddings = [generate_embeddings_batch(batch) for batch in batches]
        
        embeddings_all = []
        for embeddings in tqdm(batched_embeddings, desc=f"Generating embeddings for {len(data)} queries using {self.format_model}"):
            for embedding in embeddings:
                embeddings_all.append(embedding)
        return embeddings_all
    


