#%%
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
from os import path

#%%
# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

file_path = path.join(path.dirname(__file__), "..", "data", "HoundOfBaskerville.txt")
loader = TextLoader(file_path, encoding="utf-8")
doc = loader.load()
doc

#%%
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separators=["\n\n", "\n", " ", ".", ",", "!", "?"])
chunks = splitter.split_documents(doc)
len(chunks)

#%%