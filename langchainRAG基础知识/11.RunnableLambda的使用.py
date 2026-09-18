#RunnableLambda可以实现链中自定义函数的功能,用法:RunnableLambda(匿名函数)
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

str_parser = StrOutputParser ()#输出的是字符串
json_parser = JsonOutputParser ()#JsonOutputParser接收的是Aimessage,输出的是字典形式
my_func = RunnableLambda(lambda ai_message:{"name":ai_message.content})#匿名函数格式:lambda 参数:表达式(要拿参数做什么)
model = ChatTongyi(model="qwen3-max",streaming=True)
first_prompt = ChatPromptTemplate.from_messages([
    "我姓{firstname},我的孩子的性别是{gender},请给我的孩子取个名字,要求只给出名字就可以."
])
second_prompt = ChatPromptTemplate.from_messages([
    "请给我解析一下{name}这个名字."
])
#第一种方法
chain = first_prompt | model | my_func | second_prompt | model | str_parser
for chunk in chain.stream({"firstname": "赵", "gender": "女儿"}):
    print(chunk,end="",flush=True)

#第二种方法
chain = first_prompt | model | (lambda ai_message:{"name":ai_message.content}) | second_prompt | model | str_parser
for chunk in chain.stream({"firstname": "赵", "gender": "女儿"}):
    print(chunk,end="",flush=True)

