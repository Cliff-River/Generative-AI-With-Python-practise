#%% packages
from sentence_transformers import SentenceTransformer
import numpy as np
import seaborn as sns

#%% model
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
model = SentenceTransformer(model_name)

sentences = [
    'The cat lounged lazily on the warm windowsill.',
    'A feline relaxed comfortably on the sun-soaked ledge.',
    'The kitty reclined peacefully on the heated window perch.',
    'Quantum mechanics challenges our understanding of reality.',
    'The chef expertly julienned the carrots for the salad.',
    'The vibrant flowers bloomed in the garden.',
    'Las flores vibrantes florecieron en el jardín. ',
    'Die lebhaften Blumen blühten im Garten.'
]

#%%
sentences_embeddings = model.encode(sentences)
sentences_embeddings

#%% Correlation matrix
sentences_embeddings_corr = np.corrcoef(sentences_embeddings)
sentences_embeddings_corr

#%% visualization
# Cut off sentences for better visualization
sentences_cutted = [s if len(s) < 30 else s[:27] + '...' for s in sentences]
sns.heatmap(sentences_embeddings_corr, annot=True, xticklabels=sentences_cutted, yticklabels=sentences_cutted)

#%%