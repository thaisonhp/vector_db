import os
from typing import List, Dict
from dotenv import load_dotenv
from utils.pasering.parser import MarkItDownParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    SparseIndexParams,
    KeywordIndexParams,
)
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import models
from core.config import settings, client
from uuid import uuid4

load_dotenv()


class IndexingPipeline:
    def __init__(self):
        self.parser = MarkItDownParser()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50, length_function=len, add_start_index=True
        )
        # Dense embedding
        self.embedding_model = HuggingFaceEmbeddings(model_name=settings.embed_model_name)
        # Sparse embedding: instance đúng kiểu
        self.sparse = FastEmbedSparse(model_name=settings.sparse_model_name)
        self.vector_store = None
        print("[INIT] embedding_model:", self.embedding_model, "type:", type(self.embedding_model))
        print("[INIT] sparse_embedding:", self.sparse, "type:", type(self.sparse))

        self.client = client
        self.collection_name = settings.collection_name

    def process_markdown_file(self, file_path: str):
        parsed = self.parser.parse(file_path)
        print(f"[DEBUG] Parsed text length: {len(parsed['text'])}")

        docs = self.text_splitter.create_documents([parsed["text"]], metadatas=[{"source": file_path, **parsed["metadata"]}])
        print(f"[DEBUG] Created {len(docs)} chunks")

        # kiểm tra collection Qdrant hiện có
        exists = self.client.collection_exists(self.collection_name)
        print(f"[DEBUG] Collection exists? → {exists}")

        if not exists:
            dim = self.embedding_model.embed_query("test").shape[-1]
            print(f"[DEBUG] embedding dimension: {dim}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    settings.dense_vector_name: VectorParams(size=dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={settings.sparse_vector_name: SparseVectorParams()},
            )
            qdrant = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding_model,
                sparse_embedding=self.sparse,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name=settings.dense_vector_name,
                sparse_vector_name=settings.sparse_vector_name,
            )
        else:
            qdrant = QdrantVectorStore.from_existing_collection(
                embedding=self.embedding_model,
                sparse_embedding=self.sparse,
                collection_name=self.collection_name,
                url=settings.qdrant_url,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name=settings.dense_vector_name,
                sparse_vector_name=settings.sparse_vector_name,
            )
           
        # log verify types trước khi add
        print("[DEBUG] Before add_documents:")
        # print("  embedding type:", type(qdrant.embedding))
        # print("  sparse_embedding type:", type(qdrant.sparse_embedding))
        print("  vector_name:", qdrant.vector_name)
        print("  sparse_vector_name:", qdrant.sparse_vector_name)

        uuids = [str(uuid4()) for _ in docs]
        try:
            qdrant.add_documents(documents=docs, ids=uuids)
        except Exception as e:
            print("[ERROR] add_documents failed:", e)
            raise

        self.vector_store = qdrant
        print(f"[INFO] Successfully indexed {len(docs)} docs into '{self.collection_name}'")
        return len(docs)

    def search(self, query: str, limit: int = 5):
        qdrant = QdrantVectorStore.from_existing_collection(
                embedding=self.embedding_model,
                sparse_embedding=self.sparse,
                collection_name=self.collection_name,
                url=settings.qdrant_url,
                retrieval_mode=RetrievalMode.HYBRID,
                vector_name=settings.dense_vector_name,
                sparse_vector_name=settings.sparse_vector_name,
            )
        self.vector_store = qdrant
        if not self.vector_store:
            raise ValueError("Vector store is not initialized. Please index documents first.")

        print(f"[DEBUG] Searching for query: {query} with limit: {limit}")
        # Sử dụng vector_store để tìm kiếm
        result = self.vector_store.similarity_search_with_score(query, k=limit, filter=None)
        return result