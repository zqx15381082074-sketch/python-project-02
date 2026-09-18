#要用jq语法:"."表示根,"[]"表示数组,如果是[]代表数组里的所有元素,如果[1]表示数组的第二个元素
from langchain_community.document_loaders import JSONLoader
#测试1
loader = JSONLoader(
    file_path="../data/data1.json",
    jq_schema=".other.addr",#如果表示"跳",则表示为.hobby.[1]
)
data = loader.load()
print(data)
#测试2
loader = JSONLoader(
    file_path="../data/data2.json",
    jq_schema=".[].name",#先.[]代表数组的所有元素,再.name表示所有元素的name属性
)

data = loader.load()
print(data)
#测试3
loader = JSONLoader(
    file_path="../data/data3.jsonl",
    jq_schema=".name",
    text_content=False,#***默认True,含义是抽取的是不是字符串,如果要抽取的数据是字典形式的话,必须将其设为False
    json_lines=True,#***默认False,是否为json line,json lines就是一行一行独立的json文件组成的文件
)
data = loader.load()
print(data)