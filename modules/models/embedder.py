# embedder.py
from typing import List
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    VectorParams,
    PointStruct,
    Prefetch,
    FusionQuery,
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
from chunker import Chunk
from core.config import settings
from qdrant_client.models import (
    PointStruct,
    Prefetch,
    FusionQuery,
    Document,
    Filter,
    FieldCondition,
    MatchValue,
)


class EmbedIndexer:
    def __init__(
        self,
        collection_name: str = settings.collection_name,
        embed_model: str = settings.embed_model_name,
    ):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embedder = SentenceTransformer(embed_model)
        self.collection_name = collection_name
        self.dense_model = settings.embed_model_name
        self.sparse_model = settings.sparse_model_name
        self.dense_model_name = settings.dense_model_name
        self.sparse_model_name = settings.sparse_model_name

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.dense_model_name: VectorParams(
                        size=self.client.get_embedding_size(self.dense_model_name),
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    self.sparse_model_name: models.SparseVectorParams()
                },
            )

            print(
                f"Collection '{collection_name}' created with dense model '{self.dense_model_name}' and sparse model '{self.sparse_model_name}'."
            )
        else:
            print(
                f"Collection '{collection_name}' already exists. Using existing configuration."
            )
        self.collection_name = collection_name

    def index_chunks(self, chunks: List[Chunk], tenant_id: str = "common"):
        texts = [c.text for c in chunks]
        vecs = self.embedder.encode(texts, show_progress_bar=False)

        points = []
        # duyet casc chunk va va tao PointStruct
        for idx, (c, v) in enumerate(zip(chunks, vecs)):
            uid = hash(f"{tenant_id}-{c.file}-{idx}") & ((1 << 63) - 1)
            points.append(
                PointStruct(
                    id=uid,
                    vector=v.tolist(),
                    payload={
                        "tenant_id": tenant_id,
                        "source_file": c.file,
                        "heading": c.heading,
                        "text": c.text[:200],
                    },
                )
            )
            print(
                f"Indexing chunk {idx + 1}/{len(chunks)}: {c.text[:50]}..."
            )  # Debugging line to check chunk content
        # Upsert points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name, points=points, wait=True
        )
        print(
            f"Upserted {len(points)} points to Qdrant."
        )  # Debugging line to check number of points indexed
        if len(points) == 0:
            raise ValueError(
                "No points were indexed. Please check the chunking and embedding logic."
            )
        return len(points)

    def search(self, query: str, tenant_id: str = "common", limit: int = 5):
        # Build tenant filter
        fl = Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        )

        # Prepare hybrid prefetch: dense embedding and sparse token vector
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=Document(text=query, model=self.dense_model),
                    using=self.dense_model,
                    limit=limit * 2,
                ),
                Prefetch(
                    query=Document(text=query, model=self.sparse_model),
                    using=self.sparse_model,
                    limit=limit * 2,
                ),
            ],
            query=FusionQuery(fusion=models.Fusion.RRF),
            query_filter=fl,
            limit=limit,
            with_payload=True,
        ).points

        # Return payload metadata + optionally score
        metadata = [point.payload for point in search_result]
        return metadata
