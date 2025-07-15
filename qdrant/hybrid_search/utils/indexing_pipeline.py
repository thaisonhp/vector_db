# indexing_pipeline.py
from .pasering.parser import MarkdownParser
from .chunking.chunker import MarkdownChunker, Chunk
from .embeding.embedder import EmbedIndexer
from typing import List
from core.config import settings

class IndexingPipeline:
    def __init__(self,
                 qdrant_url: str,
                 api_key: str = None,
                 collection_name: str = settings.collection_name,
                 embed_model: str = settings.embed_model_name):
        self.parser = MarkdownParser()
        self.chunker = MarkdownChunker()
        self.indexer = EmbedIndexer(
                                    collection_name=collection_name,
                                    embed_model=embed_model)

    def index_file(self, filepath: str, tenant_id: str = "common") -> int:
        md = self.parser.convert_file(filepath)
        chunks: List[Chunk] = self.chunker.chunk(md, source_file=filepath)
        count = self.indexer.index_chunks(chunks, tenant_id=tenant_id)
        return count

    def search(self, query: str, tenant_id: str = "common", limit: int = 5):
        return self.indexer.search(query=query, tenant_id=tenant_id, limit=limit)
