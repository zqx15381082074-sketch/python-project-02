#RunnableWithMessageHistory在原有链基础上提供了带有历史会话功能的新链
#InMemoryChatMessageHistory为历史记录提供内存存储
from langchain_community.chat_models import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

str_parser = StrOutputParser ()
store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory ()#这样让每一个id对应一个内存存储会话的记忆
    return store[session_id]
model = ChatTongyi (model="qwen3-max")
prompt = ChatPromptTemplate.from_messages ([
    ("system","你是一个智能对话机器人"),
    MessagesPlaceholder("history"),
    ("human","请回答我的问题:{input}")
])
chain = prompt | model | str_parser
conversation_chain = RunnableWithMessageHistory(
    chain,
    get_history,#获取历史记录的函数
    input_messages_key="input",#输入的占位符
    history_messages_key="history"#历史对话的占位符
)
session_config = {"configurable": {"session_id": "user1"}}#这个是预先填入配置参数,运行时将作为参数传入,只要使用 LangChain 官方组件(可被 Runnable 动态读取、可配置的业务参数) ，就严格遵守 {"configurable": {...}} 格式。
print(conversation_chain.invoke({"input":"我家有一只狗"},session_config))#如果用增强链的话,传入的invoke大部分必须要用字典形式,因为有多个参数
print(conversation_chain.invoke({"input":"还有一只猫"},session_config))
print(conversation_chain.invoke({"input":"我家有几只宠物"},session_config))
#执行过程:先根据session_config查到id,再根据id通过get_history函数获取当前会话的 InMemoryChatMessageHistory对象（存放所有历史对话）,并取出历史会话并填入history里面,再将input的信息传入prompt,传给model得出答案,自动把【本次用户输入 + AI返回答案】追加写入InMemoryChatMessageHistory对象