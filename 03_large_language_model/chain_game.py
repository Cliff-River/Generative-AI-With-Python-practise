#%% Packages
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from dotenv import find_dotenv, load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from langchain_core.prompts import ChatPromptTemplate

console = Console()
load_dotenv(find_dotenv())

#%% Prepare LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
# %% Session history
store = {}
def get_session_history(session_id : str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
#%%
initial_prompt = ChatPromptTemplate.from_messages({
    ("system" , "You are a creative story teller. You will tell a story one sentence at a time. Each time I say 'continue', you will add another sentence to the story. If I say 'end', you will end the story with a conclusion."),
    ("user" , "Let's start a new story. The initial scene is {scene}, continue the story")
})
chain = initial_prompt | llm
config = { "session_id" : "default" }
llm_with_history = RunnableWithMessageHistory(chain, get_session_history=get_session_history)

context = llm_with_history.invoke({ "scene" : "a dark forest" }, config=config)
console.print(Markdown(context.content))

#%%
def process_player_choice(player_choice : str):
    response = llm_with_history.invoke({ 
        "user" : player_choice,
    }, config=config)
    return response

while True:
    player_choice = console.input("[bold green]Your turn (type 'continue' to add to the story, 'end' to finish): [/]")
    player_choice = player_choice.strip().lower()
    context = process_player_choice(player_choice)
    console.print(Markdown(context.content))
    if player_choice == "end":
        break