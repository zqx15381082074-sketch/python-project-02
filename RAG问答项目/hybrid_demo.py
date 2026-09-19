"""
混合检索对比演示: 同一个问题, 分别用三种方式检索, 直观看到差异
    1. 纯向量检索(原来的方案, 单路语义)
    2. 纯BM25(关键词)
    3. 混合检索(两路召回 + RRF融合 + 重排)
"""
from hybrid_retrieval import get_hybrid_retriever


def show(title, docs):
    print(f"\n===== {title} =====")
    if not docs:
        print("(无结果)")
        return
    for i, d in enumerate(docs, 1):
        text = d.page_content.replace("\n", " ")[:80]
        print(f"top{i}: {text}")


if __name__ == "__main__":
    query = input("请输入测试问题(回车用默认): ").strip() or "我体重120斤,适合多大码"
    hybrid = get_hybrid_retriever()

    # 1. 纯向量(语义)召回
    show("纯向量检索(语义, 原方案)", hybrid._vector_recall(query))
    # 2. 纯BM25(关键词)召回
    show("纯BM25检索(关键词)", hybrid._bm25_recall(query))
    # 3. 混合检索(双路召回+RRF融合+重排)
    show("混合检索(双路+融合+重排, 新方案)", hybrid.retrieve(query))
