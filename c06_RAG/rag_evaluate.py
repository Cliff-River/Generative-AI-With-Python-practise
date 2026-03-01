# %% packates
from datasets import Dataset
from ragas.metrics import _answer_relevancy, _context_precision, _faithfulness
from ragas import evaluate
from langchain_openai import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# %%
my_sample = {
    "question": "What is the capital of Germany in 1960?",
    "answer": [ "In 1960, the capital of Germany was Bonn. East Berlin was the capital of East Germany." ],
    "contexts": [
        [
            "Berlin is the capital of Germany.", 
            "Between 1949 and 1990, East Berlin was the capital of East Germany.", 
            "Bonn was the capital of West Germany during the same period."
        ]
    ],
    "ground_truth": ["Berlin"]
}

dataset = Dataset.from_dict(my_sample)

# %% evaluate
model = OpenAI(model="gpt-4o-mini")
metrics = [_answer_relevancy, _context_precision, _faithfulness]
res = evaluate(dataset, llm=model, metrics=metrics)
res

# %%