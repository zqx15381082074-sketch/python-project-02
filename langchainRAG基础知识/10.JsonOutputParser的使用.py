from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
str_parser = StrOutputParser ()#输出的是字符串
json_parser = JsonOutputParser ()#JsonOutputParser接收的是Aimessage,输出的是字典形式
model = ChatTongyi(model="qwen3-max",streaming=True)
first_prompt = ChatPromptTemplate.from_messages([
    "我姓{firstname},我的孩子的性别是{gender},请给我的孩子取个名字."
    "要求生成JSON形式,key是name,value是生成的名字."
])
second_prompt = ChatPromptTemplate.from_messages([
    "请给我解析一下{name}这个名字."
])
chain = first_prompt | model | json_parser | second_prompt | model | str_parser#因为prompt提示词要求输入的是JSON格式,所以需要用JsonOutputParser
for chunk in chain.stream({"firstname": "赵", "gender": "女儿"}):
    print(chunk,end="",flush=True)
