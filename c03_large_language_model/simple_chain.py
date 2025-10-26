#%%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv(find_dotenv())
#%%

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You're a AI assistant this help translating English to anthor language."),
    ("user", "Translate this sentence: '{input}' into {target_language}")
])
model = ChatOpenAI(model="openai/gpt-oss-120b", temperature=0,  base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY"))
chain = prompt_template | model | StrOutputParser()
prompt_template

#%%
chain.invoke({
    "input" : "I love programming",
    "target_language" : "Chinese"
})
#%%