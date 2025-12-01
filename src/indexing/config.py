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