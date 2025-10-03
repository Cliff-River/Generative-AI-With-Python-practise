#%%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())
#%%

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You're a AI assistant this help translating English to anthor language."),
    ("user", "Translate this sentence: '{input}' into {target_language}")
])
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt_template | model | StrOutputParser()
prompt_template

#%%
chain.invoke({
    "input" : "I love programming",
    "target_language" : "Chinese"
})
#%%