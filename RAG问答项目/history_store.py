import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, message_to_dict



class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, storage_path: str):
        self.session_id = session_id
        self.storage_path = storage_path
        # 拼接完整文件路径
        self.file_path = os.path.join(self.storage_path, self.session_id)
        # 不存在则自动创建它所在的文件夹,存在就不做这个了
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    @property#property是将方法伪装成实例属性
    def messages(self) -> list[BaseMessage]:#获得历史消息组成的消息对象列表
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                messages_data = json.load(f)
            return messages_from_dict(messages_data)#是将字典转为消息对象
        except FileNotFoundError:
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:#sequence就是字典,列表之类的将元素组合的形式,BaseMessage是ai'message,humanmessage等都是basemessage
        # 读取历史消息
        all_messages = list(self.messages)#用上面的messages方法改成的属性获得历史消息组成的列表,再用list复制一份,避免修改原数据
        # 追加新消息
        all_messages.extend(messages)
        # 对象转为字典序列化
        serialized_messages = [message_to_dict(msg) for msg in all_messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized_messages, f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        # 清空聊天记录，写入空数组
        with open(self.file_path, "w", encoding="utf-8") as f:#如果是w的话,原来有的东西会被覆盖
            json.dump([], f, ensure_ascii=False, indent=2)

def get_history(session_id):
    return FileChatMessageHistory (session_id, "../chat_history")