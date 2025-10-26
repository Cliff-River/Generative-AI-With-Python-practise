#%%
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
from os import path
import seaborn as sns

#%%
# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

file_path = path.join(path.dirname(__file__), "..", "data", "HoundOfBaskerville.txt")
loader = TextLoader(file_path, encoding="utf-8")
doc = loader.load()
doc

#%%
splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50, separators=["\n\n", "\n", " ", ".", ",", "!", "?"])
chunks = splitter.split_documents(doc)
len(chunks)

#%%
chunk_lengths = [len(chunk.page_content) for chunk in chunks]

sns.histplot(chunk_lengths, bins=50, binrange=(100, 300), kde=True)
plt.title("Chunk 长度分布图")
plt.xlabel("Chunk 长度")
plt.ylabel("数量")
plt.show()

#%%