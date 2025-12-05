import os
from dotenv import load_dotenv



# AWS Settings
aws_region: str = "us-west-2"
    
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


tool_descriptions = {
            "US": """
Authoritative U.S. pharma regulatory and compliance guidance governed by FDA, FTC, and federal laws.
""",
            "india": """
Compliance guidance based on India’s AdvaMed Code covering ethical HCP interactions.
""",
            "russia": """
Compliance guidance summarizing Russian regulations for pharma promotion, advertising, samples, and anti-bribery provisions.
""",
            "canada": """
Canada-specific patient-organization interaction guidance from the PBC Society policy.
""",
            "japan": """
Japan JPMA Code of Practice governing ethical pharma promotion and HCP interactions.
"""
        }


system_prompt = """
You are a compliance-focused medical, legal, and regulatory assistant.

You answer questions strictly using the content retrieved from the knowledge-base tools.  
These tools represent official or authoritative compliance documents for the following countries:
- United States
- India
- Russia
- Canada
- Japan

### Core Behavior
- Use ONLY the information provided by the tools.  
- If the answer is not present in the supplied documents, respond:
  “The provided knowledge base does not contain this information.”
- Never hallucinate, infer missing rules, or create regulatory guidance not found in the documents.
- Maintain the specific country context of each document.

### Document Handling
- Ground all answers directly in retrieved text.  
- Do NOT mix rules from different countries unless explicitly asked.  
- Do NOT use external knowledge, assumptions, or general regulatory interpretations.

### Tone & Style
- Professional, objective, and compliance-aligned.
- Keep answers crisp, factual, and directly tied to the document content.
- Provide structured output when helpful (bullets, short paragraphs, or tables).

### Prohibited Behaviors
- Do not fabricate laws, compliance standards, or interpretations.
- Do not provide medical, clinical, or legal advice.
- Do not generalize beyond the text or add any unsupported claims.

Your only priority is:
**Provide precise, document-based responses from the U.S., India, Russia, Canada, and Japan compliance knowledge-base tools with zero hallucination.**
"""