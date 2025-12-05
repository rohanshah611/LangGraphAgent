import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock


#from index_config import IndexConfig, get_index_config

class BedrockClientManager:
    
    def __init__(self, aws_region: str):
        self.client = boto3.client(service_name = "bedrock-runtime", region_name=aws_region)

    def get_bedrock_embeddings_llm(self, embedding_model: str, region: str = None, service_name: str = "bedrock-runtime") -> BedrockEmbeddings:

        return BedrockEmbeddings(
            client=self.client,
            model_id=embedding_model
        )
    
    def get_bedrock_agent_llm(self, region, bedrock_model_id, temperature, max_tokens, system_prompt) -> ChatBedrock:
    
        
        # Base kwargs
        kwargs = {
            "client": self.client,
            "model_id": bedrock_model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt
        }
        
        # Add guardrails if configured

        return ChatBedrock(**kwargs)