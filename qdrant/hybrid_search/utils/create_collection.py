import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    KeywordIndexParams,
)
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams, models
# from core.config import settings
load_dotenv()

client = QdrantClient(url= os.getenv("QDRANT_URL", "http://localhost:6333"))

# dense_vector_name = "dense"
# sparse_vector_name = "sparse"
# dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
# sparse_model_name = "prithivida/Splade_PP_en_v1"
if not client.collection_exists("startups"):
    client.create_collection(
        collection_name="startups",
        vectors_config={
            os.getenv("DENSE_VECTOR_NAME", "dense"): models.VectorParams(
                size=client.get_embedding_size(os.getenv("DENSE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")), 
                distance=models.Distance.COSINE
            )
        },  # size and distance are model dependent
        sparse_vectors_config={os.getenv("SPARSE_VECTOR_NAME", "sparse"): models.SparseVectorParams()},
    )