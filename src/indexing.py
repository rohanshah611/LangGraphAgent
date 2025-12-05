import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import logging
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


class PineconeManagement:
    
    def __init__(self, api_key: str):
        self.pc = Pinecone(api_key=api_key)

    #utility function 
    def index_exists(self, index_name: str) -> bool:
        return index_name in self.pc.list_indexes().names()
    #utility function 
    def namespace_exists(self, index_name: str, namespace: str) -> bool:
        idx = self.pc.Index(index_name)
        stats = idx.describe_index_stats()
        return namespace in stats.get("namespaces", {})

    def create_index(self, index_name: str, dimension: int, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1") -> None:
        
        if not self.index_exists(index_name):
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        else: return "Index already exists"
    
    # Create index content (embedding + upsert)
    
    def upsert_content(self,chunks: list,index_name: str,namespace: str, embedding_model) -> None:
   

        if not self.index_exists(index_name):
            return "Index does not exist"

        if self.namespace_exists(index_name, namespace):
            return "Namespace already exists, try a new name for namespace"


        vector_store = PineconeVectorStore.from_documents(
           documents=chunks,  
           embedding=embedding_model,
           index_name=index_name,
           namespace=namespace
           )