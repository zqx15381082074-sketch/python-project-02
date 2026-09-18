from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path="../data/data1.pdf",
    mode="single"#默认是page模式,就是按页取,而single是直接弄成一个document
    #password="123456"#这个是对于一些加密的要输密码
                     )
documents = loader.load()
print(documents)