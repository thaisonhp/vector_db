import json
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from core.config import settings , client
import tqdm


payload_path = settings.payload_path
documents = []
metadata = []

with open(payload_path) as fd:
    for line in fd:
        obj = json.loads(line)
        description = obj["description"]
        dense_document = models.Document(text=description, model=settings.dense_model_name)
        sparse_document = models.Document(text=description, model=settings.sparse_model_name)
        documents.append(
            {
                settings.dense_vector_name: dense_document,
                settings.sparse_vector_name: sparse_document,
            }
        )
        metadata.append(obj)
        client.upload_collection(
        collection_name="startups",
        vectors=tqdm.tqdm(documents),
        payload=metadata,
        parallel=4,  # Use 4 CPU cores to encode data.
        # This will spawn a model per process, which might be memory expensive
        # Make sure that your system does not use swap, and reduce the amount
        # # of processes if it does. 
        # Otherwise, it might significantly slow down the process.
        # Requires wrapping code into if __name__ == '__main__' block
    )