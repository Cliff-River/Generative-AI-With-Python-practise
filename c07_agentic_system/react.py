# %% packages
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %%
llm = ChatGroq(model="llama-3.3-70b-versatile")
tools = [TavilySearchResults(max_results=2)]
memory = MemorySaver()
agent_executer = create_react_agent(model=llm, tools=tools, checkpointer=memory)

# %%
config = {"configurable": {"thread_id": "abcd123"}}
agent_execute = agent_executer.invoke({"messages": [("user", "My name is Bert Gollnick, I am a trainer and data scientist. I live in Hamburg")]}, config=config)
print(agent_execute)

# %%
def get_latest_messages(memory : MemorySaver, config : dict):
    return memory.get_tuple(config=config).checkpoint["channel_values"]["messages"][-1].model_dump()["content"]

latest_messages = get_latest_messages(memory, config)
print(latest_messages)

# %%
agent_execute = agent_executer.invoke({"messages": [("user", "What is my name and in which country do I live?")]}, config=config)
latest_messages = get_latest_messages(memory, config)
print(latest_messages)

# %%
agent_execute = agent_executer.invoke({"messages": [("user", "What can you find about me in the internet?")]}, config=config)
latest_messages = get_latest_messages(memory, config)
print(latest_messages)

# %%
list(memory.list(config=config))

# %%
latest_messages = get_latest_messages(memory, config)
print(latest_messages)

# %%