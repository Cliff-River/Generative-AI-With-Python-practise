import pandas as pd
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

model_name = "dslim/bert-base-NER"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

text = "My name is Cliff, coming from China, I have a dog called Dolly."
print(pd.DataFrame(nlp(text)))