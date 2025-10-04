#%%
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())
#%%
#%% Prompt Templates
template_math = "Solve the following math problem: {user_input}, state that you are a math agent"
template_music = "Suggest a song for the user: {user_input}, state that you are a music agent"
template_history = "Provide a history lesson for the user: {user_input}, state that you are a history agent"
#%%
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()
# %% Math-Chain
prompt_math = ChatPromptTemplate.from_messages([
    ("system", template_math),
    ("human", "{user_input}")
])
chain_math = prompt_math | model | StrOutputParser()
# %% Music-Chain
prompt_music = ChatPromptTemplate.from_messages([
    ("system", template_music),
    ("human", "{user_input}")
])
chain_music = prompt_music | model | StrOutputParser()
#%% 
# History-Chain
prompt_history = ChatPromptTemplate.from_messages([
    ("system", template_history),
    ("human", "{user_input}")
])
chain_history = prompt_history | model | StrOutputParser()

chains = [chain_math , chain_music, chain_history]
embeddings = OpenAIEmbeddings()
chain_embeddings = embeddings.embed_documents(["math", "music", "history"])

len(chain_embeddings)
#%%