from utils import BedrockClientManager
from tools import PineconeRetrieverTool
from config import aws_region, embedding_model, pinecone_index_name, system_prompt, agent_model, temperature, max_tokens, tool_name, tool_description, pinecone_namespace
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
memory = MemorySaver()

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

from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))


#Create Local streamlit app

### Title of the app
st.title("Enhanced NASA Chatbot With AWS Bedrock")

## MAin interface for user input
st.write("Ask any question")
user_input=st.text_input("You:")

if user_input:
    config={"configurable":{"thread_id":"1"}}
    state = {"messages": [HumanMessage(content=user_input)]}
    result = graph.invoke(state,config=config)
    st.write("\n✅ Final Answer:\n", result["messages"][-1].content)
else:
    st.write("Please provide the user input")



