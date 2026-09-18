#chatprompttemplate可以实现提示词里有多个角色,所以主要用的是这个

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt_template = ChatPromptTemplate.from_messages([
    ("system","你是一个边塞诗人"),
    MessagesPlaceholder("history"),#messageplaceholder是占位的作用,并且生成一个key名字叫 history
    ("human","请根据上面的对话,仿写一段文字")
])
model = ChatTongyi(model="qwen3-max",streaming=True)
history_data = [
    ("human","写一首诗"),
    ("ai","白发三千丈，高挂云间。"),
]
prompt = prompt_template.invoke({"history":history_data})#用invoke渲染提示词模板,是key:value的形式,形成完整的提示词
res = model.invoke(prompt)#将提示词传入模型
print(res.content)
#messageplaceholder只能用invoke调用
#静态方法✅ from_messages：造对象用（流水线，开工前使用）
#实例方法✅ prompt.invoke()：对象造好之后，让对象干活（实例方法，开工后使用）
