
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi

from langchain_core.tools import tool


@tool(description="查询天气")
def get_wheather():
    return "明天是晴天"
agent = create_agent(
    model=ChatTongyi(model="qwen3-max",streaming=True),
    tools=[get_wheather],
    system_prompt="请使用工具查询天气，并返回结果",
)
#1.直接输出
# res = agent.invoke(
#     {"messages":[{"role":"user","content":"明天的天气如何"}]}
# )
# for msg in res["messages"]:
#     print(type(msg).__name__,msg.content)#type(msg).__name__是获取类的类名
#2.流式输出
res = agent.stream(
    {"messages":[{"role":"user","content":"明天的天气如何"}]},
    stream_mode="values"
)
for chunk in res:
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(type(latest_message).__name__,latest_message.content)
    try:
        if latest_message.tool_calls:
            print(f"调用了工具:{[tool['name'] for tool in latest_message.tool_calls]}")
    except AttributeError:
        pass