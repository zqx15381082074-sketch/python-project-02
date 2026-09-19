from langchain_classic.docstore import document
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory

from history_store import get_history
from vector_store_service import VectorStoreService
from hybrid_retrieval import HybridRetriever


class RagService:
    def __init__(self):
        self.vector_store = VectorStoreService(embedding=DashScopeEmbeddings())
        # 原来是单路向量检索: self.retriever = self.vector_store.get_retriever()
        # 现在升级为混合检索: BM25关键词 + 向量语义 双路召回 -> RRF融合 -> 重排
        self.retriever = HybridRetriever(self.vector_store.vector_store)
        self.model = ChatTongyi(model="qwen3-max",streaming=True)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system","你是一个智能对话机器人,要求根据以下资料回答问题,资料是:{context}"),
            ("system","并且根据以下历史会话回答"),
            MessagesPlaceholder("history"),
            ("human","请回答我的问题:{input}")
        ])

    def get_chain(self):

        def my_func1(documents:list[Document]):#将list[document]转为字符串
            if not documents:
                return "不存在context"
            my_str = ""
            for document in documents:
                my_str += document.page_content
            return my_str
        def my_func2(value):
            return value["input"]
        def mu_func3(value):
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value
        chain = {"input":RunnablePassthrough(),"context":RunnableLambda(my_func2)| RunnableLambda(lambda q: self.retriever.retrieve(q)) | RunnableLambda(my_func1)} | RunnableLambda(mu_func3)| self.prompt | self.model | StrOutputParser ()

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )
        return conversation_chain
if __name__ == "__main__":
    rag_service = RagService()
    res = rag_service.get_chain().invoke({"input":"我体重120斤,适合多大码"},config={"configurable": {"session_id": "user_001"}})#用增强链有多个参数,必须要用字典传参
    print(res)