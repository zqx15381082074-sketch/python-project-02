from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
example_template = PromptTemplate.from_template("{word}的反义词是{antonym}")
example_data = [
    {"word": "好", "antonym": "坏"},
    {"word": "大", "antonym": "小"},
    {"word": "高", "antonym": "低"},
]#列表内置字典形式
few_shot_template = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data,
    prefix="给你几个反义词的示例如下",#这是示例之前的提示词
    suffix="基于前面几个示例,给我生成{input_word}的反义词",#这是示例之后的提示词
    input_variables=["input_word"],#声明在前缀或者后缀中需要用到的变量,列表形式
)
prompt_text = few_shot_template.invoke(input={"input_word":"动"}).to_string()#形成补充完变量之后的完整prompt字符串形式
model = ChatTongyi(model="qwen3-max")
res = model.invoke(input=prompt_text)#将提示词喂给大模型
print(res)