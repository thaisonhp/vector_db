from utils.processor.indexer import Indexer


class RetrievalPipeline:
    def __init__(self):
        self.indexer = Indexer()

    def retrieval(self, query: str , limit: int = 5):
        return self.indexer.search(query=query, limit=limit)