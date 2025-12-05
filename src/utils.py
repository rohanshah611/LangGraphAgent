import boto3
from langchain_aws import BedrockEmbeddings, ChatBedrock


#from index_config import IndexConfig, get_index_config

class BedrockClientManager:
    
    def __init__(self, aws_region: str):
        self.client = boto3.client(service_name = "bedrock-runtime", region_name=aws_region)

    def get_bedrock_embeddings_llm(self, embedding_model: str) -> BedrockEmbeddings:
        print('embedding invoked')
        return BedrockEmbeddings(
            client=self.client,
            model_id=embedding_model
        )
    
    def get_bedrock_agent_llm(self, bedrock_model_id: str, temperature: float, max_tokens: int, system_prompt: str) -> ChatBedrock:
        print("llm invoked")
        return ChatBedrock(client = self.client,model_id = bedrock_model_id,temperature = temperature,max_tokens = max_tokens,system_prompt = system_prompt)
