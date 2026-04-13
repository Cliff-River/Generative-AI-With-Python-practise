# %% Packages
import os
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AnyMessage
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
from IPython.display import display, Image

load_dotenv(find_dotenv())

# %%
llm = ChatOpenAI(model="openai/gpt-4o-mini", base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))

# %% Function to count how many times a character appears in a word
def count_char_in_word(word: str, char: str) -> str:
    """Counts how many times a character appears in a word."""
    c = word.count(char)
    return f"The character '{char}' appears {c} times in the word '{word}'."

# Test function count_char_in_word
test_result = count_char_in_word("hello", "l")
test_result

# %%
llm_with_tools = llm.bind_tools([count_char_in_word])

tool_call = llm_with_tools.invoke(["user", "How many times does the character 'l' appear in the word 'hello'?"])
pprint(tool_call)

# %%
tool_call.additional_kwargs["tool_calls"]

#%%
class MessageState(TypedDict):
    messages: list[AnyMessage]

def tool_calling_llm(state : MessageState) -> MessageState:
    return { "messages" : [llm_with_tools.invoke(state["messages"])] }

builder = StateGraph(MessageState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_node("tools", ToolNode([count_char_in_word]))
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()

#%%
display(Image(graph.get_graph().draw_mermaid_png()))

# %%
initial_state = {"messages" : [HumanMessage(content="How many times does the character 'l' appear in the word 'hello'?")]}
final_state = graph.invoke(initial_state)
for n in final_state["messages"]:
    print(n.pretty_print())

# %%