#%%
import os
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(find_dotenv())
#%%
#%% Prompt Templates
template_math = "Solve the following math problem: {user_input}, state that you are a math agent"
template_music = "Suggest a song for the user: {user_input}, state that you are a music agent"
template_history = "Provide a history lesson for the user: {user_input}, state that you are a history agent"
#%%
model = ChatOpenAI(model="gemini-2.5-flash", temperature=0, base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.environ.get("GEMINI_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
chain_embeddings = embeddings.embed_documents(["math", "music", "history"])

len(chain_embeddings)

#%%
def my_prompt_router(input):
    input_embeddings = embeddings.embed_query(input)
    scores = cosine_similarity([input_embeddings], chain_embeddings)
    most_simular_index = scores.argmax()
    return chains[most_simular_index]

#%%
query = "Who composed the moonlight sonata?"
chain = my_prompt_router(query)
print(chain.invoke(query))

#%%