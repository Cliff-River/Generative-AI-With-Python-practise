from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

model_name = "deepset/roberta-base-squad2"
nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)

QA_input = {
    'question': 'What are the benefits of remote work?',
    'context': 'Remote work allows employees to work from anywhere, providing flexibility and a better work-life balance. It reduces commuting time, lowers operational costs for companies, and can increase productivity for self-motivated workers.'
}

print(nlp(QA_input))
print("***************************")

#%%
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(dir(tokenizer))
