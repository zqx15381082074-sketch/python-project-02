import streamlit as st
from rag_service import RagService
st.title("智能对话助手")
st.divider ()
if "message" not in st.session_state:
    st.session_state["message"] = [{"role":"assistant","content":"你好,我是智能对话助手,请开始你的对话吧!"}]
if "service" not in st.session_state:
    st.session_state["service"] = RagService()
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])
prompt = st.chat_input("请输入问题")
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role":"user","content":prompt})
    response = st.session_state["service"].get_chain().stream({"input": prompt}, config={"configurable": {"session_id": "user1"}})
    response_message = st.empty()
    full_response = ""
    for chunk in response:
        full_response += chunk
        response_message.write(full_response)
    st.session_state["message"].append({"role":"assistant","content":full_response})
