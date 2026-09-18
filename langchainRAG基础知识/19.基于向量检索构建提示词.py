from itertools import chain

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore

model = ChatTongyi (model="qwen3-max")
prompt = ChatPromptTemplate.from_messages ([
    ("system","你是一个智能对话机器人,要求根据以下资料回答问题,资料是:{context}"),
    ("human","请回答我的问题:{input}")
])
vector_store = InMemoryVectorStore(
    embedding=DashScopeEmbeddings()
)
vector_store.add_texts(#add_texts可以直接讲文本转为document再转为向量,省去了TEXTloader的步骤,多个document的话是list[document]
    ["我叫小王,今年18岁,来自中国,现在在杭州"]
)
input_text = "介绍一下你自己."
res = vector_store.similarity_search(
    query=input_text,
    k=2
)#根据已有的数据存储做出回答,回答的是一个list[document]
reference_text = ""
for chunk in res:#将document转为字符串,以便传给提示词模板
    reference_text += chunk.page_content#.page_content是Document里的两个属性之一,表示文本字符串内容,而metadata是字典形式的表示配置信息
chain = prompt | model | StrOutputParser()
print(chain.invoke({"input":input_text,"context":reference_text}))
