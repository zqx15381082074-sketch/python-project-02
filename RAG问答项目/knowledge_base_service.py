import datetime
import os
import config_data as config
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib

def check_md5(md5_str: str):
    if not os.path.exists(config.md5_path):
        open(config.md5_path, "w", encoding="utf-8").close()#创建文件
        return False
    else:
        for line in open(config.md5_path, "r", encoding="utf-8").readlines():#readlines是[1\n,2\n,3]这种形式组成的列表
            line = line.strip()#strip是去掉字符串前后的空格以及回车
            if line == md5_str:
                return True
        return  False
def save_md5(md5_str: str):
    with open(config.md5_path,"a",encoding="utf-8")as f:
        f.write(md5_str+"\n")
def get_string_md5(input_str: str):#将字符串转为md5字符串
    str_bytes = input_str.encode("utf-8")#将字符串转为字节
    md5_obj = hashlib.md5()#创建md5对象
    md5_obj.update(str_bytes)#只能输入字节,所以要先转为字节
    md5_hex = md5_obj.hexdigest()#转为十六进制md5字符串,便于观看
    return md5_hex

#这个类的load_by_str的逻辑是把传入的大片文件的内容转为向量存入数据库中,并且保存md5值
class KnowledgeBaseService:
    def __init__(self):
       self.chroma = Chroma(#外部向量数据库
         collection_name="test",
         embedding_function=DashScopeEmbeddings(),
         persist_directory="data/chroma"
       )
       self.spliter = RecursiveCharacterTextSplitter(#text分割器
         chunk_size=500,
         chunk_overlap=50,
         separators=["\n\n", "\n", " ", "."],
         length_function=len
       )
    def load_by_str(self,data:str,filename):#将传入的字符串转为向量存储在向量数据库,并保存md5的数据(为了方便不用每次查询如果存在的话还要查询数据库,太麻烦)
           md5_hex = get_string_md5(data)
           if check_md5(md5_hex):#如果有了,就在这一步停止,后面的不做处理
               return ["已存在"]
           if len(data) >= 500:
               data_spliter = self.spliter.split_text(data)
           else:
               data_spliter = [data]#因为split_text返回的是一个列表,所以这里要转为列表
           metadata = {
               "filename": filename,
               "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "operator":"admin"
           }
           self.chroma.add_texts(#因为chroma里存的是先document再转为向量,所以这里text相当于.page_content,还要加上metadata
               texts=data_spliter,
               metadatas=[metadata for _ in data_spliter]#要给每一个document添加一个metadata,所以这里要循环
           )
           save_md5(md5_hex)
           return ["上传成功"]
