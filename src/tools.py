from langchain_core.tools import Tool
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever
from pinecone import Pinecone
from langchain_aws import BedrockEmbeddings


class PineconeRetrieverTool:

    def __init__(self, embeddings:BedrockEmbeddings, namespace:str, pinecone_api_key:str, pinecone_index_name:str, tool_name:str, tool_description:str):
        """
        Args:
            
        """
        self.embeddings = embeddings
        self.namespace = namespace
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.pinecone_index_name = pinecone_index_name
        self.tool_name = tool_name
        self.tool_description = tool_description

        
    def create_retrievers(self, text_key="text") -> VectorStoreRetriever:

        vectorstore = PineconeVectorStore(
            index = self.pinecone_index_name,
            embedding = self.embeddings,
            namespace = self.namespace,
            text_key = text_key
        )
        retriever = vectorstore.as_retriever()
        retriever.name = self.tool_name
        return retriever
    
    
    def create_retriever_function(self, query: str, ) -> str:
        retriver = self.create_retrievers()
        print("Using retriver "+ retriver.name)
        docs = retriver.invoke(query)
        return "\n".join([doc.page_content for doc in docs])
    
    def create_retriver_tool(self):
        return Tool(name=self.tool_name, description=self.tool_description,func=self.create_retriever_function)

