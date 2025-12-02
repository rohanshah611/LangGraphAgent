import os
from dotenv import load_dotenv



# AWS Settings
aws_region: str = "us-east-1"
    
# Bedrock Settings
bedrock_model: str = "us.amazon.nova-pro-v1:0"
embedding_model: str = "amazon.titan-embed-text-v2:0"
temperature: float = 0.0
max_tokens: int = 2048

# Guardrails 
#TBD

# Pinecone Settings
pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
pinecone_index_name: str = "medical-compliance"
namespaces: dict = {
                "US": "US",
                "india": "india",
                "russia": "russia",
                "canada": "canada",
                "japan": "japan"
            }


#Indexing 

INDEX_CONFIG = {
    "documents_path": "/path/to/knowledge_base",

    "embedding_model": "text-embedding-3-large",
    
    "index_name": "medical-rag-index",
    "namespace_name": "us-regulatory",

    "dimensions": 3072,
    "metric": "cosine",

    "chunk_size": 500,
    "chunk_overlap": 50,

    # If namespace exists:
    # True  -> skip indexing completely
    # False -> reindex (delete namespace -> recreate -> index)
    "skip_if_namespace_exists": True
}