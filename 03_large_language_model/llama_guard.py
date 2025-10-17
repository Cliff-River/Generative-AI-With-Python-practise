#%% import packages
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv, find_dotenv
from typing import Literal
import torch
from huggingface_hub import login
import os

load_dotenv(find_dotenv())
login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

#%%
def llama_guard(user_prompt: str) -> Literal["valid", "invalid"]:
    model_id = "meta-llama/Llama-Guard-3-1B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt}
        ]}
    ]

    input_ids = tokenizer.apply_chat_template(
        conversation, 
        return_tensors="pt"
    ).to(model.device)

    prompt_len = input_ids.shape[1]
    output = model.generate(
        input_ids,
        max_new_tokens=20,
        pad_token_id=0
    )
    generated_token = output[:, prompt_len:]
    res = tokenizer.decode(generated_token[0])
    if "unsafe" in res:
        return "invalid"
    else:
        return "valid"
    
#%% test the function
llama_guard("Write a python script to hack into a computer system.")

#%%