from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model, before_model, after_agent, before_agent, wrap_model_call, \
    wrap_tool_call
from langchain_community.chat_models import ChatTongyi

from langchain_core.tools import tool
from langgraph.runtime import Runtime


@before_agent
def log_before_agent(state: AgentState, runtime: Runtime) -> None:
    # agent执行前会调用这个函数并传入state和runtime两个对象
    print(f"[before agent]agent启动，并附带{len(state['messages'])}消息")


@after_agent
def log_after_agent(state: AgentState, runtime: Runtime) -> None:
    print(f"[after agent]agent结束，并附带{len(state['messages'])}消息")


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[before_model]模型即将调用，并附带{len(state['messages'])}消息")


@after_model
def log_after_model(state: AgentState, runtime: Runtime) -> None:
    print(f"[after_model]模型调用结束，并附带{len(state['messages'])}消息")

@wrap_model_call
def model_call_hook(request, handler):
    print("模型调用啦")
    return handler(request)


@wrap_tool_call
def monitor_tool(request, handler):
    print(f"工具执行：{request.tool_call['name']}")
    print(f"工具执行传入参数：{request.tool_call['args']}")
    return handler(request)

@tool(description="查询天气")
def get_wheather():
    return "明天是晴天"
agent = create_agent(
    model=ChatTongyi(model="qwen3-max",streaming=True),
    tools=[get_wheather],
    middleware=[log_before_agent, log_after_agent, log_before_model, log_after_model, model_call_hook, monitor_tool]
)

res = agent.invoke(
    {"messages":[{"role":"user","content":"明天的天气如何"}]}
)
for msg in res["messages"]:
    print(type(msg).__name__,msg.content)