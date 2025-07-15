from pydantic_settings import BaseSettings
from typing import Optional, List
from qdrant_client import QdrantClient , models
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import os 
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("COLLECTION_NAME")
    embedding_dimension: int = os.getenv("EMBEDDING_DIMENSION", 128)
    distance_metric: str        = os.getenv("DISTANCE_METRIC", "COSINE")
    metadata: Optional[dict] = os.getenv("METADATA", None)
    dense_vector_name: str = os.getenv("DENSE_VECTOR_NAME", "dense")
    sparse_vector_name: str = os.getenv("SPARSE_VECTOR_NAME", "sparse")
    dense_model_name : str = os.getenv("DENSE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    sparse_model_name : str = os.getenv("SPARSE_MODEL_NAME", "prithivida/Splade_PP_en_v1")
    payload_path: Optional[str] = os.getenv("PAYLOAD_PATH", "qdrant/hybrid_search/data/startups_demo.json")
    embed_model_name: str = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
# khoi tao QdrantClient 
client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
if not client.collection_exists("startups"):
    client.create_collection(
        collection_name="startups",
        vectors_config={
            settings.dense_model_name: models.VectorParams(
                size=client.get_embedding_size(settings.dense_model_name), 
                distance=models.Distance.COSINE
            )
        },  # size and distance are model dependent
        sparse_vectors_config={settings.sparse_model_name: models.SparseVectorParams()},
    )
