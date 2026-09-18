from dashscope.cli import embeddings
from langchain_community.embeddings import DashScopeEmbeddings
# 创建模型对象 不传model默认用的是 text-embeddings-v1
model = DashScopeEmbeddings()
print(model.embed_query("我喜欢你"))#将字符串单次转换为向量
print(model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))#批量转换

from langchain_ollama import OllamaEmbeddings
model = OllamaEmbeddings(model="qwen3-embedding:4b")
print(model.embed_query("我喜欢你"))#将字符串单次转换为向量
print(model.embed_documents(["我喜欢你", "我稀饭你", "晚上吃啥"]))#批量转换
