from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from pathlib import Path


class ChunkingManager:
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: list):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def recursive_text_splitter(self, path: str, metadata: dict) -> List[str]:
        docs = PyMuPDFLoader(path).load()

        for doc in docs:
            doc.metadata.update(metadata)
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators
        )

        return splitter.split_documents(docs)
    
    
    

    
    