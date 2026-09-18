#以下方法是将各种形式的数据转换成document(知识库文档的标准对象)
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(#创建了对象以及配置信息
    file_path="../data/data1.csv",
    csv_args={
        "delimiter": ",",#代表用逗号进行分隔
        "quotechar": '"',#代表被分隔符包围的一个总体是单引号还是双引号
        # "fieldnames": ["name", "age", "city","hobby"],这个fieldname是在没有表头的时候使用的,如果有的话,会将这个作为表头,原表头内容会变成第一条数据
    },
    encoding="utf-8"
)     #此时数据已经从csv文件读取成document形式了
#第一种方法
documents = loader.load()#真正读取数据
print(documents)
#第二种方法
for document in loader.lazy_load():
    print(document)
#这两种方法都生成的是list[document],但是第一个是全部读取写进内存,第二种是迭代器,一个一个读取
