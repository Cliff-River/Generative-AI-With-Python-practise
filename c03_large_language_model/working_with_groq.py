#%%
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# %%
load_dotenv("../.env")
# %%

model_name = "llama-3.3-70b-versatile"
model = ChatGroq(model=model_name, temperature=0.5, api_key=os.getenv("GROQ_API_KEY"))
#%%

res = model.invoke("张三打断了李四的腿, 张三和李四的关系是？")
res.dict()

#%%
res.model_dump()["content"]

