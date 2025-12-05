from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingManager:
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def recursive_chunking(self, path: str) -> List[str]:
        docs = PyMuPDFLoader(path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )

        return splitter.split_documents(docs)