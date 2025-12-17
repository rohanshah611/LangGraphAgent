from utils import BedrockClientManager
from tools import PineconeRetrieverTool
from config import aws_region, embedding_model, pinecone_index_name, system_prompt, agent_model, temperature, max_tokens, tool_name, tool_description, pinecone_namespace, memory_id
from dotenv import load_dotenv
load_dotenv() 
import os
from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langchain.agents import create_agent
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph_checkpoint_aws import AgentCoreMemorySaver, AgentCoreMemoryStore

app = BedrockAgentCoreApp()

# Initialize 
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
bedrock_client = BedrockClientManager(aws_region = aws_region)
br_embedding = bedrock_client.get_bedrock_embeddings_llm(embedding_model)
agent_llm = bedrock_client.get_bedrock_agent_llm(bedrock_model_id = agent_model, temperature = temperature, max_tokens = max_tokens, system_prompt = system_prompt, guardrail_config = None)


#Create retrivers: 
retriever = PineconeRetrieverTool(embeddings = br_embedding, namespace = pinecone_namespace, pinecone_api_key = PINECONE_API_KEY, pinecone_index_name = pinecone_index_name, tool_name = tool_name, tool_description = tool_description)
retriever_tool = retriever.create_retriver_tool()
tools = [retriever_tool]
llm_with_tool=agent_llm.bind_tools(tools)


#Create Agent: 
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

#Initialize Memory
#memory = MemorySaver()
memory = AgentCoreMemorySaver(memory_id=memory_id)

#Create LLM Node
def tool_calling_llm(state:AgentState):
    return {"messages":[llm_with_tool.invoke(state["messages"])]}

## Grpah
builder=StateGraph(AgentState)
builder.add_node("tool_calling_llm",tool_calling_llm)
builder.add_node("tools",ToolNode(tools))

## Add Edges
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    tools_condition
)
builder.add_edge("tools","tool_calling_llm")

## compile the graph
graph=builder.compile(checkpointer=memory)


@app.entrypoint
def agent_invocation(payload, context):
    """Handler for agent invocation in AgentCore runtime with memory support"""
    print("Received payload:", payload)
    print("Context:", context)
    
    # Extract query from payload
    query = payload.get("prompt", "No prompt found in input")
    
    # Extract or generate actor_id and thread_id
    actor_id = payload.get("actor_id", "default-user")
    thread_id = payload.get("thread_id", payload.get("session_id", "default-session"))
    
    # Configure memory context
    config = {
        "configurable": {
            "thread_id": thread_id,  # Maps to AgentCore session_id
            "actor_id": actor_id     # Maps to AgentCore actor_id
        }
    }
    state = {"messages": [HumanMessage(content=query)]}
    # Invoke the agent with memory
    result = graph.invoke(state,config=config)
    
    # Extract the final answer from the result
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else "No response generated"
    
    # Return the answer
    return {
        "result": answer,
        "actor_id": actor_id,
        "thread_id": thread_id
    }


if __name__ == "__main__":
    app.run()



