import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings
from config import aws_region, bedrock_model, embedding_model, temperature, max_tokens, pinecone_api_key, pinecone_index_name, namespaces

def get_bedrock_client(service_name: str = "bedrock-runtime", region: str = None):
    """
    Get AWS Bedrock client
    
    Args:
        service_name: AWS service name
        region: AWS region
        
    Returns:
        Boto3 client
    """
    if region is None:
        region = aws_region
    
    return boto3.client(service_name, region_name=region)


def get_bedrock_embeddings(embedding_model: str, region: str = None) -> BedrockEmbeddings:
    """
    Get Bedrock embeddings model
    
    Args:
        embedding_model: Embedding model name available in AWS
        region: AWS region
        
    Returns:
        BedrockEmbeddings instance
    """
    client = get_bedrock_client("bedrock-runtime", region)
    return BedrockEmbeddings(
        client=client,
        model_id = embedding_model
    )
