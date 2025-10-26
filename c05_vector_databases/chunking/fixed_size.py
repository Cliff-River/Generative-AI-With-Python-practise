#%%
from os import path
import matplotlib.pyplot as plt
import numpy as np
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from pprint import pprint

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

directory_path = path.join(path.dirname(__file__), "..", "data")
loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})

#%%
docs  = loader.load()
docs[1].page_content

#%%
splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=50, separator=" ")
chunks = splitter.split_documents(docs)
len(chunks)

#%%
pprint(chunks[10].page_content)
print("-------------------------------")
pprint(chunks[11].page_content)

#%%
# 计算每个chunk的大小（字符数）
chunk_sizes = [len(chunk.page_content) for chunk in chunks]

# 打印基本统计信息
print(f"总chunk数量: {len(chunk_sizes)}")
print(f"最小chunk大小: {min(chunk_sizes)} 字符")
print(f"最大chunk大小: {max(chunk_sizes)} 字符")
print(f"平均chunk大小: {np.mean(chunk_sizes):.2f} 字符")
print(f"中位数chunk大小: {np.median(chunk_sizes)} 字符")

#%% 绘制chunk大小分布直方图
plt.figure(figsize=(12, 6))
plt.hist(chunk_sizes, bins=50, range=(100, 300), alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(chunk_sizes), color='red', linestyle='dashed', linewidth=2, label=f'平均值: {np.mean(chunk_sizes):.2f}')
plt.axvline(np.median(chunk_sizes), color='green', linestyle='dashed', linewidth=2, label=f'中位数: {np.median(chunk_sizes):.2f}')
plt.axvline(256, color='orange', linestyle='solid', linewidth=2, label='目标大小: 256')
plt.title('Chunk大小分布直方图')
plt.xlabel('Chunk大小（字符数）')
plt.ylabel('频率')
# 设置x轴刻度以匹配直方图范围
plt.xticks(range(100, 350, 50))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

#%% 绘制箱线图以显示分布情况
plt.figure(figsize=(10, 6))
box_plot = plt.boxplot(chunk_sizes, patch_artist=True, labels=['Chunk大小'])
plt.setp(box_plot['boxes'], facecolor='skyblue')
plt.setp(box_plot['medians'], color='red')
plt.title('Chunk大小分布箱线图')
plt.ylabel('大小（字符数）')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%%