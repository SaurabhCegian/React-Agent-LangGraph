from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_message
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_message]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

tools = [add]

model = ChatOpenAI(model="gpt-4o", bind_tools=tools)

def model_call(state:AgentState):
    system_prompt = SystemMessage(content="You are my AI Assistant, Please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

