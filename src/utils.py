import boto3
from langchain_aws import BedrockEmbeddings

#from index_config import IndexConfig, get_index_config

class BedrockClientManager:
    
    def __init__(self, aws_region: str, embedding_model: str):
        self.aws_region = aws_region

   
    def get_bedrock_client(self, service_name: str = "bedrock-runtime", region: str = None):
        if region is None:
            region = self.aws_region

        return boto3.client(service_name, region_name=region)


    def get_bedrock_embeddings(self, embedding_model: str, region: str = None) -> BedrockEmbeddings:
        client = self.get_bedrock_client("bedrock-runtime", region)

        return BedrockEmbeddings(
            client=client,
            model_id=embedding_model
        )