from re import search

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma



class VectorStoreService:
    def __init__(self,embedding):
        self.embedding = embedding
        self.vector_store = Chroma(
            collection_name="test",
            embedding_function=self.embedding,
            persist_directory="data/chroma"
        )
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": 1})
if __name__ == "__main__":
   res = VectorStoreService(embedding=DashScopeEmbeddings()).get_retriever()
   print(res.invoke("我是男生腰围104,适合多大码?"))#这个retriever的invoke传的不是字典了,他override成了字符串
