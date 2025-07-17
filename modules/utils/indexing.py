from utils.processor.indexer import Indexer


class IndexingPipeline:
    def __init__(self , collection_name: str = None):
        self.indexer = Indexer(collection_name=collection_name)

    def add_file(self, file_path: str):
        self.indexer.indexing(file_path)