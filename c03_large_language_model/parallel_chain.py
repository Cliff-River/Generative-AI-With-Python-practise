#%%
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv(find_dotenv())
#%%
polite_prompt = ChatPromptTemplate.from_messages([
    ("system" , "You're a polite assistant, replay in a friently and polite mamer"),
    ("human", "{topic}")
])
savage_prompt = ChatPromptTemplate.from_messages([
    ("system" , "You're a savage assistant, replay in a savage and angry mamer"),
    ("human", "{topic}")
])
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
polite_chain = polite_prompt | model | StrOutputParser()
savage_chain = savage_prompt | model | StrOutputParser()
map_chain = RunnableParallel({
    "polite" : polite_chain,
    "savage" : savage_chain
})
result = map_chain.invoke({
    "topic" : "生活的意义是什么?"
})
print(result)

#%%