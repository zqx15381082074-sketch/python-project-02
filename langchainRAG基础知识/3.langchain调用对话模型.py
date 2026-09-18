# # 1.调用对话模型
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, content

model = ChatTongyi(model="qwen3-max",streaming=True)
messages = [
    SystemMessage(content="你是一个边塞诗人"),
    HumanMessage(content="写一首诗"),
    AIMessage(content="白发三千丈，高挂云间。"),
    HumanMessage(content="根据上面内容,再做一首诗")
]
#或者可以简写为列表里面套元组的形式:
# messages = [
#     ("system","你是一个边塞诗人"),
#     ("human","写一首诗"),
#     ("ai","白发三千丈，高挂云间。"),
#     ("human","根据上面内容,再做一首诗")
# ]
#而且还可以是("system","你是一个{role}")变量占位的形式
res = model.stream(input=messages)
for chunk in res:
    print(chunk.content,end="",flush=True)#.content是因为这里的chunk是一套东西,要抽出仅聊天的内容
# 2.调用本地ollama
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, content

model = ChatOllama(model="deepseek-r1:7b",streaming=True)
messages = [
    SystemMessage(content="你是一个边塞诗人"),
    HumanMessage(content="写一首诗"),
    AIMessage(content="白发三千丈，高挂云间。"),
    HumanMessage(content="根据上面内容,再做一首诗")
]
res = model.stream(input=messages)
for chunk in res:
    print(chunk.content,end="",flush=True)