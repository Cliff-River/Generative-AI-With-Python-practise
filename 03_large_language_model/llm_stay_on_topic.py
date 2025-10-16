#%%
from typing import Literal
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())

#%%
classifier = pipeline("zero-shot-classification",  model="facebook/bart-large-mnli")

def guard_medical_prompt(user_prompt: str) -> Literal["valid", "invalid"]:
    candidate_labels = ["politics", "finance", "technology", "healthcare", "sports"]
    result = classifier(user_prompt, candidate_labels=candidate_labels)
    if result["labels"][0] == "healthcare":
        return "valid"
    else:
        return "invalid"

#%% TEST guard_medical_prompt
user_prompt = "Should I buy stocks of Apple, Google, or Amazon?"  # invalid
guard_medical_prompt(user_prompt)

#%%
def guard_chain(user_input: str) -> str:
    if guard_medical_prompt(user_input) == "invalid":
        return "I'm sorry, I can only answer questions related to healthcare."
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that can answer questions about healthcare.")
        ("user", "{input}")
    ])
    model = ChatGroq(model="llama3-8b-8192", temperature=0.5)
    chain = prompt_template | model | StrOutputParser()
    return chain.invoke({"input": user_input})

#%% TEST guard_chain
guard_chain(user_prompt)

#%%
