#%% packages
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import find_dotenv, load_dotenv
from pprint import pprint

load_dotenv(find_dotenv())
#%%
prompt = hub.pull("hardkothari/prompt-maker")
prompt.input_variables
#%%
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# %% chain
chain = prompt | model | StrOutputParser()

lazy_prompt = "summer, vacation, beach"
task = "Shakespeare poem"
improved_prompt = chain.invoke({"lazy_prompt": lazy_prompt, "task": task})
pprint(improved_prompt)
#%%
print("=======================================================")
res = model.invoke(improved_prompt)
pprint(res.content)

#%%