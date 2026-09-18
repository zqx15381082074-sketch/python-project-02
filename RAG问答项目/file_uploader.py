import os,sys
import streamlit as st#st是模块,不是类,里面定义了很多函数,可以直接调用

sys.path.append(os.path.dirname(__file__))
from knowledge_base_service import KnowledgeBaseService

st.title("知识库更新服务")
uploader_file = st.file_uploader(
    "请上传知识库文件",
    type=["txt"],
    accept_multiple_files=False,
    key="up1",
    help="上传知识库文件，支持pdf和txt格式"
)
if "service" not in st.session_state:#session_state存储的内容是字典形式
    st.session_state["service"] = KnowledgeBaseService()#这样防止每次上传都会创建一个service对象,开销大
if uploader_file is not None:
    file_name = uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size/1024
    st.subheader(f"文件名:{file_name}")
    st.write(f"文件类型:{file_type}")
    st.write(f"文件大小:{file_size:.2f}KB")
    text = uploader_file.getvalue().decode('utf-8')#decode是字节转字符串,解码,getvalue是获得文件字节内容
    st.write(f"文件内容:{text}")
    with st.spinner("正在上传..."):
        res = st.session_state["service"].load_by_str(text,file_name)#存入外部数据库中
    st.write(res)



