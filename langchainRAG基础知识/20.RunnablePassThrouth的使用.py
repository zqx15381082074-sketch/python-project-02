#这部分是将上一节的基于向量检索也加入chain
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

model = ChatTongyi (model="qwen3-max")
prompt = ChatPromptTemplate.from_messages ([
    ("system","你是一个智能对话机器人,要求根据以下资料回答问题,资料是:{context}"),
    ("human","请回答我的问题:{input}")
])
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings()
)
def format_func(documents):#将list[document]转为字符串,以便传给提示词模板
    str = ""
    for document in documents:
        str += document.page_content
    return str

vector_store.add_texts(#add_texts可以直接讲文本转为document再转为向量,省去了TEXTloader的步骤,多个document的话是list[document]
    ["通过跑步,打篮球,游泳可以减肥哦"]
)
input_text = "如何减肥"
retirever = vector_store.as_retriever(search_kwargs={"k": 2})#作用是返回一个runnable接口的子类实例对象,从而可以加入chain,as_retriever后面再用invoke相当于similarity_search
chain = {"input": RunnablePassthrough(), "context": retirever | format_func}  | prompt | model | StrOutputParser()
res = chain.invoke({"input": input_text})
print(res)
#chain的注意事项:
#1.RunnablePassThrough将本来传进的第一个字典的第一个嵌套链,也就是retriever的input_text给复制一份,从而形成完整字典以便传入prompt
#2.如format_func这样的函数在传入链中的时候一定不要加(),如果加的话就是首先执行函数了
