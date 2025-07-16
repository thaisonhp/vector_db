from utils.processor.indexer import Indexer


class IndexingPipeline:
    def __init__(self):
        self.indexer = Indexer()

    def add_file(self, file_path: str):
        self.indexer.indexing(file_path)