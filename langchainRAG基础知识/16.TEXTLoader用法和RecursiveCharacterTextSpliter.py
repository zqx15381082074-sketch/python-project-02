#text里面装的是字符串
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(
    "../data/data1.text",
    encoding="utf-8"
)
document = loader.load()#它只能返回仅有一个document对象的list,所以后续要用Recursive......
splitter = RecursiveCharacterTextSplitter(
    chunk_size=50,#分段的最大字符数
    chunk_overlap=20,#分段可重叠的最大字符数
    separators=["\n\n", "\n", " ", "."],#分段的依据,有这些东西就分段
    length_function=len#每段字符统计的函数
)
split_documents = splitter.split_documents(document)
print(split_documents)