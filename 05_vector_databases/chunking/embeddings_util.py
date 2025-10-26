"""
免费嵌入模型工具模块

本模块封装了 Hugging Face 免费嵌入模型的使用，
作为无需 API 密钥的 OpenAIEmbeddings 替代方案。
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

def get_free_embeddings(model_name="all-MiniLM-L6-v2"):
    """
    获取免费的嵌入模型实例
    
    Args:
        model_name (str): HuggingFace模型名称，默认为"all-MiniLM-L6-v2"
        
    Returns:
        HuggingFaceEmbeddings: 嵌入模型实例
    """
    return HuggingFaceEmbeddings(model_name=model_name)

# 默认使用的嵌入模型实例
DEFAULT_EMBEDDINGS = get_free_embeddings()