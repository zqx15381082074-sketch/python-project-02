from langchain_community.llms.tongyi import Tongyi
llm = Tongyi(model="qwen-max")
res = llm.stream("一加一等于几")
for chunk in res:
    print(chunk,end="",flush=True)

from langchain_ollama import OllamaLLM
model = OllamaLLM(model="deepseek-r1:7b")
res = model.stream(input="一加一等于几")
for chunk in res:
    print(chunk,end="",flush=True)