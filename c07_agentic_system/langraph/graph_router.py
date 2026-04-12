# %% packages
import random
from IPython.display import Image, display
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from rich.markdown import Markdown
from rich.console import Console
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %%
llm = ChatGroq(model="moonshotai/kimi-k2-instruct")

class State(TypedDict):
    graph_state : dict[str, str | dict[str, str]]

def node_router(state: State):
    topic = state["graph_state"].get("topic", "No topic provided")
    state["graph_state"]["processed_topic"] = topic
    print("User-provided topic:", topic)
    return { "graph_state": state["graph_state"] }

def node_pro(state : State):
    topic = state["graph_state"]["topic"]
    response = llm.invoke(f"Generate arguments in favor of {topic}. Answer in bullet points. Max 5 words per bullet point.")
    state["graph_state"]["result"] = { "side" : "pro", "argument": response }
    return { "graph_state": state["graph_state"] }

def node_contra(state : State):
    topic = state["graph_state"]["topic"]
    response = llm.invoke(f"Generate arguments against {topic}. ")
    state["graph_state"]["result"] = { "side" : "contra", "argument": response }
    return { "graph_state": state["graph_state"] }

def edge_pro_or_contra(state: State):
    decision = random.choice(["node_pro", "node_contra"])
    state["graph_state"]["decision"] = decision
    print("Routing to:", decision)
    return decision

builder = StateGraph(State)
builder.add_node("node_router", node_router)
builder.add_node("node_pro", node_pro)
builder.add_node("node_contra", node_contra)
builder.add_edge(START, "node_router")
builder.add_conditional_edges("node_router", edge_pro_or_contra)
builder.add_edge("node_pro", END)
builder.add_edge("node_contra", END)

# %%
graph = builder.compile()
display(Image(graph.get_graph().draw_mermaid_png()))

# %%