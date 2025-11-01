# %%
import gensim.downloader as api
import random
import seaborn.objects as so
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %% Load google word vectors
word_vectors = api.load("word2vec-google-news-300")

# %% Get the size of the word vector
studied_word = "mathematics"
print(word_vectors[studied_word].shape)

# %%
word_vectors[studied_word]

# %%
sims = word_vectors.most_similar(studied_word)
similar_words = [n[0] for n in sims[:5]]
similar_words

# %% Get random word for word vectors
number_of_random = 20
random.seed(42)
# word_vectors.key_to_index can also work.
random_words = random.sample(list(word_vectors.key_to_index.keys()), number_of_random)
random_words

# %%
word_to_plot = random_words + similar_words
embeddings = np.array([])
for word in word_to_plot:
    embeddings = (
        np.vstack((embeddings, word_vectors[word]))
        if embeddings.size
        else word_vectors[word]
    )
embeddings.shape

# %%
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
df = pd.DataFrame(embeddings_2d, columns=["x", "y"])
df["word"] = word_to_plot
df["color"] = ["random"] * number_of_random + ["similar"] * len(similar_words)
df.shape

# %% Plot the words
(so.Plot(df, x="x", y="y", color="color", text="word")
    .add(so.Dots())
    .add(so.Text())
)

#%% Visualizing it via lines
df_arithmetic = pd.DataFrame({
    "word": ['paris', 'germany', 'france', 'berlin', 'madrid', 'spain']
})
pca = PCA(n_components=2)
embedding_arithmetic = np.array([])
for word in df_arithmetic['word']:
    embedding_arithmetic = (
        np.vstack((embedding_arithmetic, word_vectors[word]))
        if embedding_arithmetic.size
        else word_vectors[word]
    )

# Apply PCA
arithmetic_2d = pca.fit_transform(embedding_arithmetic)
df_arithmetic['x'] = arithmetic_2d[:, 0]
df_arithmetic['y'] = arithmetic_2d[:, 1]

#%% Visualize it via pyplot with lines
plt.figure(figsize=(10, 10))
plt.scatter(df_arithmetic['x'], df_arithmetic['y'], marker="o")

# add vector from paris to france, berlin to germany, madrid to spain
plt.arrow(df_arithmetic['x'][0], df_arithmetic['y'][0],
            df_arithmetic['x'][2] - df_arithmetic['x'][0],
            df_arithmetic['y'][2] - df_arithmetic['y'][0],
            head_width=0.01, head_length=0.01, fc='r', ec='r')
plt.arrow(df_arithmetic['x'][3], df_arithmetic['y'][3],
            df_arithmetic['x'][1] - df_arithmetic['x'][3],
            df_arithmetic['y'][1] - df_arithmetic['y'][3],
            head_width=0.01, head_length=0.01, fc='r', ec='r')
plt.arrow(df_arithmetic['x'][4], df_arithmetic['y'][4],
            df_arithmetic['x'][5] - df_arithmetic['x'][4],
            df_arithmetic['y'][5] - df_arithmetic['y'][4],
            head_width=0.01, head_length=0.01, fc='r', ec='r')

# Add labels for words
for i, word in enumerate(df_arithmetic['word']):
    plt.annotate(word, (df_arithmetic['x'][i]+0.01, df_arithmetic['y'][i]+0.01))

#%%