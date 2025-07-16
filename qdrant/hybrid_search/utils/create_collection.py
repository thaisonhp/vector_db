from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    MultiVectorConfig,
    MultiVectorComparator,
    HnswConfigDiff,
    models,
)
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding

def create_hybrid_rerank_collection(
    client: QdrantClient,
    collection_name: str,
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
    late_model: LateInteractionTextEmbedding,
):
    """
    Tạo collection Qdrant cho hybrid search + rerank:
     - dense semantic vector
     - sparse BM25
     - late-interaction ColBERT
    """

    # lấy kích thước embedding từ một câu test
    # dense = [float] vector
    dense_vec = dense_model.embed("test")
    dense_vec_list = list(dense_vec)
    dense_dim = len(dense_vec_list[0])
    print("dense_dim:", dense_dim)
    # late-interaction = list of token vectors [[...]]
    late_vecs = late_model.embed("test")
    # chọn kích thước của mỗi token vector (hàng đầu tiên)
    late_dim = len(list(late_vecs)[0][0])
    print("late_interaction:", late_dim)

    # xóa collection cũ nếu tồn tại
    if client.collection_exists(collection_name):
        print(f"[⚠️] Deleting existing '{collection_name}'…")
        client.delete_collection(collection_name)

    # tạo collection mới
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
            "late_interaction": VectorParams(
                size=late_dim,
                distance=Distance.COSINE,
                multivector_config=MultiVectorConfig(
                    comparator=MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=HnswConfigDiff(m=0),  # disable HNSW for rerank
            ),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    print(
        f"✅ Created '{collection_name}' with: dense '{dense_model}' ({dense_dim}), "
        f"sparse '{sparse_model}', late '{late_model}' ({late_dim})"
    )
    return client