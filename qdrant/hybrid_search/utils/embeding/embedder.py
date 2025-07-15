# embedder.py
from typing import List
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
from utils.chunking.chunker import Chunk
from core.config import settings
from qdrant_client.models import PointStruct, Prefetch, FusionQuery, Document, Filter, FieldCondition, MatchValue

class EmbedIndexer:
    def __init__(self,
                 collection_name: str = settings.collection_name,
                 embed_model: str = settings.embed_model_name):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.embedder = SentenceTransformer(embed_model)
        self.collection_name = collection_name

        dim = self.embedder.get_sentence_embedding_dimension()
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="tenant_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
                is_tenant=True
            )

    def index_chunks(self, chunks: List[Chunk], tenant_id: str = "common"):
        texts = [c.text for c in chunks]
        vecs = self.embedder.encode(texts, show_progress_bar=False)

        points = []
        # duyet casc chunk va va tao PointStruct
        for idx, (c, v) in enumerate(zip(chunks, vecs)):
            uid = hash(f"{tenant_id}-{c.file}-{idx}") & ((1<<63)-1)
            points.append(PointStruct(
                id=uid,
                vector=v.tolist(),
                payload={
                    "tenant_id": tenant_id,
                    "source_file": c.file,
                    "heading": c.heading,
                    "text": c.text[:200]
                }
            ))
        # Upsert points to Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return len(points)

    def search(self, text: str, tenant_id: str = "common", limit: int = 5):
        # Build tenant filter
        fl = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])
        
        # Prepare hybrid prefetch: dense embedding and sparse token vector
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=Document(text=text, model=self.dense_model),
                    using=self.dense_model,
                    limit=limit * 2
                ),
                Prefetch(
                    query=Document(text=text, model=self.sparse_model),
                    using=self.sparse_model,
                    limit=limit * 2
                ),
            ],
            query=FusionQuery(fusion=models.Fusion.RRF),
            query_filter=fl,
            limit=limit,
            with_payload=True
        ).points

        # Return payload metadata + optionally score
        metadata = [point.payload for point in search_result]
        return metadata
