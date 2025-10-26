import pandas as pd
from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-uncased')
result = unmasker("My [MASK] is Cliff")
print(pd.DataFrame(result))