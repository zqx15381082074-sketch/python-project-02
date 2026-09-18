#chain就是将各组件串联在一起,上一个组件的输出作为下一个组件的输入,并且组件都要是Runnable的子类
#可以通过invoke或者stream来触发整条链的运行
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt_template = ChatPromptTemplate.from_messages([#只有chatprompttemplate后面加from_message才能实现多角色
    ("system","你是一个边塞诗人"),
    MessagesPlaceholder("history"),#messageplaceholder是占位的作用,并且生成一个key名字叫 history
    ("human","请根据上面的对话,仿写一段文字")
])
model = ChatTongyi(model="qwen3-max",streaming=True)
history_data = [
    ("human","写一首诗"),
    ("ai","白发三千丈，高挂云间。"),
]
chain = prompt_template | model
for chunk in chain.stream({"history":history_data}):
    print(chunk.content,end="",flush=True)
