import os
from typing import List, Dict
from dotenv import load_dotenv
from models.parser import MarkItDownParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
   PointStruct
)
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import models
from core.config import settings, client
from uuid import uuid4
from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from utils.manage_collection.collection_manager import CollectionManager

load_dotenv()


class Indexer:
    def __init__(self):
        self.parser = MarkItDownParser()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50, length_function=len, add_start_index=True
        )
        self.client = client
        self.collection_name = settings.collection_name

        self.collection = None  # de chon ra collection de lam viec 

        self.collection_manager = CollectionManager(self.client, self.collection_name)
        # khoi tao cac vector embedding models phuc vu cho  hybrid search + rerank
        dense_embedding_model = TextEmbedding(settings.dense_embedding_model)
        self.embedding_model = dense_embedding_model
        bm25_embedding_model = SparseTextEmbedding(settings.bm25_embedding_model)
        self.sparse_embedding_model = bm25_embedding_model
        late_interaction_embedding_model = LateInteractionTextEmbedding(
            settings.late_model_name
        )
        self.late_interaction_embedding_model = late_interaction_embedding_model
    
    def indexing(self, file_path: str):
        parsed = self.parser.parse(file_path)

        documents = self.text_splitter.create_documents(
            [parsed["text"]], metadatas=[{"source": file_path, **parsed["metadata"]}]
        )
        print(documents[:2])  # Show first 2 documents for debugging

        # embedding cac chunks
        dense_embeddings = list(self.embedding_model.embed(texts))
        bm25_embeddings = list(self.sparse_embedding_model.embed(texts))
        late_interaction_embeddings = list(self.late_interaction_embedding_model.embed(texts))


        # kiểm tra collection Qdrant hiện có
        exists = self.client.collection_exists(self.collection_name)
        print(f"[DEBUG] Collection exists? → {exists}")

        texts = [doc.page_content if isinstance(doc.page_content, str) else "" for doc in documents]
        print(f"[DEBUG] Texts extracted: {len(texts)}")
        
        print(
            f"[DEBUG] Embeddings created: {len(dense_embeddings)} dense, {len(bm25_embeddings)} sparse, {len(late_interaction_embeddings)} late-interaction"
        )
        if not exists:
            self.collection = self.collection_manager.create_hybrid_rerank_collection(
                client=self.client,
                collection_name=self.collection_name,
                dense_model=self.embedding_model,
                sparse_model=self.sparse_embedding_model,
                late_model=self.late_interaction_embedding_model,
            )
            print(f"[INFO] Created new collection '{self.collection_name}'")
        else:
           self.collection = self.collection_manager.get_collection(
               collection_name=self.collection_name
           )
           print(f"[INFO] Using existing collection '{self.collection_name}'")
   
        # upsert points to Qdrant collection
        points = []
        for idx, (
            dense_embedding,
            bm25_embedding,
            late_interaction_embedding,
            doc,
        ) in enumerate(
            zip(
                dense_embeddings,
                bm25_embeddings,
                late_interaction_embeddings,
                documents,
            )
        ):

            point = PointStruct(
                id=idx,
                vector={
                    "dense": dense_embedding,
                    "sparse": bm25_embedding.as_object(),
                    "late_interaction": late_interaction_embedding,
                },
                payload={"document": doc},
            )
            points.append(point)
        operation_info = self.collection.upsert(
            collection_name=settings.collection_name, points=points
        )
        return operation_info

    def search(self, query: str, limit: int = 5):
        dense_vectors = next(self.embedding_model.query_embed(query))
        sparse_vectors = next(self.sparse_embedding_model.query_embed(query))
        late_vectors = next(self.late_interaction_embedding_model.query_embed(query))
        prefetch = [
            models.Prefetch(
                query=dense_vectors,
                using="dense",
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vectors.as_object()),
                using="sparse",
                limit=20,
            ),
        ]
        results = self.collection.query_points(
                collection_name=settings.collection_name,
                prefetch=prefetch,
                query=late_vectors,
                using="late_interaction",
                with_payload=True,
                limit=10,
        )

        # print(f"[DEBUG] Searching for query: {query} with limit: {limit}")
        # # Sử dụng vector_store để tìm kiếm
        # result = self.vector_store.similarity_search_with_score(
        #     query, k=limit, filter=None
        # )
        return results
