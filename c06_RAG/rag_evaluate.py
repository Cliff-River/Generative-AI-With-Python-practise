# %% packates
from datasets import Dataset
from ragas.metrics import _answer_relevancy, _context_precision, _faithfulness
from ragas import evaluate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# %%
my_sample = {
    "question": ["What is the capital of Germany in 1960?"],
    "contexts": [
        [
            "Berlin is the capital of Germany.", 
            "Between 1949 and 1990, East Berlin was the capital of East Germany.", 
            "Bonn was the capital of West Germany during the same period."
        ]
    ],
    "answer": [ "In 1960, the capital of Germany was Bonn. East Berlin was the capital of East Germany." ],
    "ground_truth": ["Berlin"]
}

dataset = Dataset.from_dict(my_sample)
# %%
metrics = [ _context_precision, _answer_relevancy, _faithfulness ]
metrics

# %% evaluate
model = ChatOpenAI(model="openai/gpt-4o-mini", base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"))
res = evaluate(dataset, llm=model, metrics=metrics)
print(res)

# %%