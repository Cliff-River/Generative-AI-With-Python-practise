# %% packages
import string
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import jieba

use_jieba = True

# %% Function to preprocess Chinese text
def preprocess_text(text: str) -> list[str]:
    """
    预处理中文文本
    
    Args:
        text (str): 输入的中文文本
        
    Returns:
        list[str]: 分词后的词语列表
    """
    # 移除标点符号和空白字符
    text = ''.join([char for char in text if char not in string.punctuation and char.strip()])
    
    # 根据是否有jieba选择分词方式
    if use_jieba:
        # 使用jieba进行分词
        return list(jieba.cut(text))
    else:
        # 使用字符级分词作为备选方案
        return list[str](text)

# %% 中文语料库
corpus = [
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。人工智能的研究领域包括机器人、语言识别、图像识别、自然语言处理和专家系统等。近年来，随着深度学习技术的发展，人工智能在各个领域取得了突破性进展。",
    "机器学习是人工智能的一个分支，是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、计算复杂性理论等多门学科。机器学习理论主要是设计和分析一些让计算机可以自动「学习」的算法。常见的机器学习方法包括监督学习、无监督学习和强化学习。",
    "北京是中国的首都，位于华北平原北部，是中国政治、文化和国际交往中心。北京有着悠久的历史和丰富的文化遗产，如故宫、长城和天坛等。同时，北京也是中国重要的科技创新中心，拥有众多高校和研究机构。",
    "上海是中国的经济中心和最大城市，位于长江三角洲地区。上海是一个国际化大都市，拥有现代化的城市景观和丰富的商业活动。外滩和浦东新区是上海的标志性地区，展现了传统与现代的交融。",
    "自然语言处理是人工智能的一个重要分支，主要研究如何让计算机理解和生成人类语言。自然语言处理技术广泛应用于机器翻译、文本摘要、情感分析、问答系统等领域。近年来，预训练语言模型的出现极大地推动了自然语言处理技术的发展。",
    "数据科学是一门结合统计学、计算机科学和领域知识的交叉学科，主要研究如何从大量数据中提取有价值的信息。数据科学的核心步骤包括数据收集、数据清洗、数据分析和数据可视化。数据科学家需要掌握编程、统计学和机器学习等技能。"
]

# 分词处理中文文本
tokenized_corpus = [preprocess_text(doc) for doc in corpus]
# 对于TF-IDF，如果使用jieba，我们需要将分词结果重新组合成空格分隔的字符串
if use_jieba:
    tokenized_corpus_tfidf = [' '.join(tokens) for tokens in tokenized_corpus]
else:
    # 否则使用完整句子
    tokenized_corpus_tfidf = corpus

# %% 初始化模型
bm25 = BM25Okapi(tokenized_corpus)
# 配置TF-IDF向量器以适应中文文本
if use_jieba:
    # 如果使用jieba分词，使用更适合中文的配置
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b')
else:
    # 否则使用字符级别的n-gram
    tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 2))

# %% 定义相似度计算函数
def calculate_similarity_scores(user_query: str):
    """
    计算给定用户查询的BM25和TF-IDF相似度分数
    
    Args:
        user_query (str): 用户的查询字符串
        
    Returns:
        dict: 包含分词后的查询和相似度分数的字典
    """
    # 对查询进行预处理
    tokenized_query_bm25 = preprocess_text(user_query)
    bm25_scores = bm25.get_scores(tokenized_query_bm25)

    # TF-IDF处理
    # 对查询进行与语料库相同的预处理
    if use_jieba:
        # 如果语料库使用了jieba分词，查询也需要分词并以空格分隔
        tokenized_query_tfidf = ' '.join(preprocess_text(user_query))
    else:
        tokenized_query_tfidf = user_query
    
    # 只在第一次调用时拟合，之后直接转换
    if not hasattr(calculate_similarity_scores, 'tfidf_matrix'):
        calculate_similarity_scores.tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_corpus_tfidf)
    
    tfidf_query_vector = tfidf_vectorizer.transform([tokenized_query_tfidf])
    cosine_similarities = cosine_similarity(tfidf_query_vector, calculate_similarity_scores.tfidf_matrix)
    tfidf_scores = cosine_similarities.tolist()[0]
    
    # 打印结果
    print("BM25 分数:", bm25_scores)
    print("分词后的查询 (BM25):", tokenized_query_bm25)
    print("处理后的查询 (TF-IDF):", tokenized_query_tfidf)
    print("TF-IDF 分数:", tfidf_scores)
    
    # 返回结果字典
    return {
        "tokenized_query_bm25": tokenized_query_bm25,
        "original_query": user_query,
        "bm25_scores": bm25_scores,
        "tfidf_scores": tfidf_scores,
        "top_document_index": np.argmax(bm25_scores)
    }

# %% 示例查询 1: 测试人工智能主题
user_query = "什么是人工智能及其应用领域？"
print("\n示例查询 1:", user_query)
results = calculate_similarity_scores(user_query)
print("最相关的文档 (BM25):", corpus[np.argmax(results["bm25_scores"])])
print("最相关的文档 (TF-IDF):", corpus[np.argmax(results["tfidf_scores"])])

# %% 示例查询 2: 测试城市主题
user_query = "上海的标志性地区"
print("\n示例查询 2:", user_query)
results = calculate_similarity_scores(user_query)
print("最相关的文档 (BM25):", corpus[np.argmax(results["bm25_scores"])])
print("最相关的文档 (TF-IDF):", corpus[np.argmax(results["tfidf_scores"])])

# %% 示例查询 3: 测试机器学习主题
user_query = "监督学习和无监督学习的区别"
print("\n示例查询 3:", user_query)
results = calculate_similarity_scores(user_query)
print("最相关的文档 (BM25):", corpus[np.argmax(results["bm25_scores"])])
print("最相关的文档 (TF-IDF):", corpus[np.argmax(results["tfidf_scores"])])

# %% 示例查询 4: 测试自然语言处理主题
user_query = "自然语言处理技术应用"
print("\n示例查询 4:", user_query)
results = calculate_similarity_scores(user_query)
print("最相关的文档 (BM25):", corpus[np.argmax(results["bm25_scores"])])
print("最相关的文档 (TF-IDF):", corpus[np.argmax(results["tfidf_scores"])])

# %% 示例查询 5: 测试数据科学主题
user_query = "数据科学的核心步骤"
print("\n示例查询 5:", user_query)
results = calculate_similarity_scores(user_query)
print("最相关的文档 (BM25):", corpus[np.argmax(results["bm25_scores"])])
print("最相关的文档 (TF-IDF):", corpus[np.argmax(results["tfidf_scores"])])

# %% 示例查询 6: 测试跨主题相关性
user_query = "北京的科技创新中心"
print("\n示例查询 6:", user_query)
results = calculate_similarity_scores(user_query)
print("最相关的文档 (BM25):", corpus[np.argmax(results["bm25_scores"])])
print("最相关的文档 (TF-IDF):", corpus[np.argmax(results["tfidf_scores"])])

# %%