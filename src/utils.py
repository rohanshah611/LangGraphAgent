import boto3
from typing import Optional, List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyMuPDFLoader
from utils import get_bedrock_client, get_bedrock_embeddings
import os
from dotenv import load_dotenv
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


#from index_config import IndexConfig, get_index_config

class BedrockClientManager:
    
    def __init__(self, aws_region: str):
        self.aws_region = aws_region

   
    def get_bedrock_client(self, service_name: str = "bedrock-runtime", region: str = None):
        if region is None:
            region = self.aws_region

        return boto3.client(service_name, region_name=region)


    def get_bedrock_embeddings(self, embedding_model: str, region: str = None) -> BedrockEmbeddings:
        """
        Return a BedrockEmbeddings instance.
        """
        client = self.get_bedrock_client("bedrock-runtime", region)

        return BedrockEmbeddings(
            client=client,
            model_id=embedding_model
        )
    


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