# %% packages
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from IPython.display import display, Image
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %%
class State(TypedDict):
    messages : Annotated[list, add_messages]

llm = ChatGroq(model="moonshotai/kimi-k2-instruct")

# %%
def assistant(state: State) -> str:
    return { "messages": [llm.invoke(state["messages"])] }

graph_builder = StateGraph(State)
graph_builder.add_node("assistant", assistant)
graph_builder.add_edge(START, "assistant")
graph_builder.add_edge("assistant", END)
graph = graph_builder.compile()

# %%
display(Image(graph.get_graph().draw_mermaid_png()))

# %%