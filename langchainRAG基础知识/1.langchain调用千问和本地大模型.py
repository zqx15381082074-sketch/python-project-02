from langchain_community.llms.tongyi import Tongyi
llm = Tongyi(model="qwen-max")
res = llm.invoke(input="一加一等于几")
print(res)

from langchain_ollama import OllamaLLM
model = OllamaLLM(model="deepseek-r1:7b")
res = model.invoke(input="一加一等于几")
print(res)

