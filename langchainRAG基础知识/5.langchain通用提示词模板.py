from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(#from_template(模板文本) → 实例化出模板对象（带空位）
    "我的邻居姓{lastname}, 刚生了{gender}, 你帮我起个名字, 简单回答。"
)
#1.调用.format方法注入信息即可
prompt_text = prompt_template.format(lastname="张", gender="女儿")#模板对象调用 .format() → 填充空位，产出完整 prompt 字符串
model = Tongyi(model="qwen-max")
res = model.invoke(input=prompt_text)
print(res)
#2.可以通过链
model = Tongyi(model="qwen-max")
chain = prompt_template | model
res = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res)