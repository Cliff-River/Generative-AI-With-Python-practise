#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

#%% function for Chain-of-Thought Prompting
def chain_of_thought_prompting(prompt: str, model_name: str = "openai/gpt-oss-20b") -> str:
    model = ChatGroq(model_name=model_name)
    prompt = ChatPromptTemplate.from_messages(messages=[
        ("system", "You are a helpful assistant and answer precise and concise."),
        ("user", f"{prompt} \n think step by step")
    ])
    # print(prompt)
    chain = prompt | model
    return chain.invoke({}).content

# %% Self-Consistency CoT
def self_consistency_cot(prompt: str, number_of_runs: int = 3) -> str:
    # run CoT multiple times
    res = []
    for _ in range(number_of_runs):
        current_res = chain_of_thought_prompting(prompt)
        print(current_res)
        res.append(current_res)
    
    # concatenate all results
    res_concat = ";".join(res)
    self_consistency_prompt = f"You will get multiple answers in <<>>, separated by ; <<{res_concat}>> Extract only the final equations and return the most common equation as it was provided originally. If there is no common equation, return the most likely equation."
    self_consistency_prompt = self_consistency_prompt.replace("{", "[").replace("}", "]")
    # self_consistency_prompt = ";".join(self_consistency_prompt)
    messages = [
        ("system", "You are a helpful assistant and answer precise and concise."),
        ("user", f"{self_consistency_prompt}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages=messages)
    model = ChatGroq(model_name="openai/gpt-oss-20b")
    chain = prompt | model
    print("----- Final Answer -----")
    return chain.invoke({}).content


#%% Test
user_prompt = "The goal of the Game of 24 is to use the four arithmetic operations (addition, subtraction, multiplication, and division) to combine four numbers and get a result of 24. The numbers are 3, 4, 6, and 8. It is mandatory to use all four numbers. Please check the final equation for correctness. Hints: Identify the basic operations, Prioritize multiplication and division, Look for combinations that make numbers divisible by 24, Consider order of operations, Use parentheses strategically, Practice with different number combinations"

# %%
# res = chain_of_thought_prompting(prompt=user_prompt)
# res
#%%
print("----- Self-Consistency CoT -----")
res = self_consistency_cot(prompt=user_prompt, number_of_runs=5)
print(res)

# %%