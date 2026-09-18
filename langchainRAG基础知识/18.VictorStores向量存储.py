from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
方法1:内存向量存储
vector_store = InMemoryVectorStore(#vector:向量
    embedding=DashScopeEmbeddings()
)
方法2:外部数据库向量存储
vector_store = Chroma(
    collection_name="test",
    embedding_function=DashScopeEmbeddings(),
    persist_directory="data/chroma"
)
loader = CSVLoader(
    file_path="../data/data2.csv",
    encoding="utf-8",
    source_column="source"
)
documents = loader.load()
#添加,这个函数会将list[Document]添加进去并自动转为向量存储
vector_store.add_documents(
    documents= documents,
    ids=["id"+str(i) for i in range(1,len(documents)+1)]#列表推导式
)
#删除
vector_store.delete(["id1","id2"])#删除向量要用列表形式
#检索:根据向量相似度最高的优先匹配,将向量转换为list[Document]形式传出
result = vector_store.similarity_search(
   "学python要注意什么",
    k=2
)
print(result)