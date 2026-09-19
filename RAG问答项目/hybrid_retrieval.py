"""
混合检索模块: BM25(关键词召回) + 向量(语义召回) 双路召回 -> RRF融合 -> 重排(rerank)

为什么需要混合检索?
    向量检索: 靠语义相似度, 擅长"换个说法也能找到"(比如"尺码"能匹配"型号"),
             但对精确词不敏感(比如"104厘米"这种数字/专有名词容易被语义淹没)
    BM25:     经典的关键词检索算法, 擅长精确匹配(词命中就加分), 但不懂同义词
    两路各召回 top8 -> RRF融合成 top5 -> 重排模型精排出 top3
    这样既有语义理解, 又不会漏掉精确关键词, 这是工业界RAG的标准做法
"""
import os
import json
import urllib.request

from rank_bm25 import BM25Okapi
import jieba
from langchain_core.documents import Document


class HybridRetriever:
    def __init__(self, vector_store, k_recall=8, k_fuse=5, k_final=3):
        """
        :param vector_store: Chroma向量库对象(不是retriever, 因为要直接调用similarity_search和get)
        :param k_recall: 每一路(BM25/向量)各自召回的候选数量
        :param k_fuse:    RRF融合后保留的数量
        :param k_final:   重排后最终返回给大模型的数量
        """
        self.vector_store = vector_store
        self.k_recall = k_recall
        self.k_fuse = k_fuse
        self.k_final = k_final
        self._docs = []   # BM25用的全量切片文档(和向量库里的切片一一对应)
        self._bm25 = None
        self._build_bm25_index()

    def _tokenize(self, text: str):
        # BM25是按"词"匹配的, 中文必须先分词(jieba), 英文天然有空格不用管
        return [w for w in jieba.lcut(text) if w.strip()]

    def _build_bm25_index(self):
        # 从Chroma拉出全量切片文档, 给BM25建索引
        # 这样保证BM25和向量库检索的是同一批切片, 融合时才可对比
        data = self.vector_store._collection.get(include=["documents", "metadatas"])
        self._docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        if self._docs:
            corpus = [self._tokenize(d.page_content) for d in self._docs]
            self._bm25 = BM25Okapi(corpus)  # BM25经典实现, 传分好词的语料即可

    def refresh(self):
        # 新文档入库后调用, 重建BM25索引(向量库是实时更新的, BM25需要重建)
        self._build_bm25_index()

    # ---------- 第一路: 向量召回(语义) ----------
    def _vector_recall(self, query: str):
        return self.vector_store.similarity_search(query, k=self.k_recall)

    # ---------- 第二路: BM25召回(关键词) ----------
    def _bm25_recall(self, query: str):
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))  # 每个切片的BM25得分
        # 按得分降序取前k个, 得分为0的(一个词都没命中)不要
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k_recall]
        return [self._docs[i] for i in top if scores[i] > 0]

    # ---------- 融合: RRF(Reciprocal Rank Fusion, 倒数排名融合) ----------
    def _rrf_fuse(self, doc_lists, k=5, k_const=60):
        """
        两路召回的分数体系完全不同(BM25是无界的词频得分, 向量是0~1的余弦相似度),
        不能直接加, 所以用RRF: 只看"排名", 不看"分数"
        公式: score = 1/(60+rank1) + 1/(60+rank2)  (rank从1开始)
        一篇文档在两路里都排前面, 融合分就高; k_const=60是原论文的默认值, 起平滑作用
        """
        pool = {}  # key: 切片文本 -> [Document对象, 累计融合分]
        for docs in doc_lists:
            for rank, doc in enumerate(docs):
                key = doc.page_content
                if key not in pool:
                    pool[key] = [doc, 0.0]
                pool[key][1] += 1.0 / (k_const + rank + 1)
        ranked = sorted(pool.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    # ---------- 重排: 调DashScope的gte-rerank模型 ----------
    def _rerank(self, query: str, docs):
        """
        召回(粗排)追求"快和全", 用便宜的方法先筛出候选;
        重排(精排)追求"准", 用交叉编码器模型逐对打分(query和文档一起送入模型)
        把融合后的top5交给重排模型, 精排出最终top3
        """
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key or not docs:
            return docs[: self.k_final]  # 没配key就直接返回融合结果
        body = json.dumps({
            "model": "gte-rerank-v2",
            "input": {"query": query, "documents": [d.page_content for d in docs]},
            "parameters": {"return_documents": False, "top_n": self.k_final},
        }).encode()
        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
            data=body,
            method="POST",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        )
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=10))
            results = resp["output"]["results"]  # [{index:候选序号, relevance_score:相关度}, ...]
            return [docs[r["index"]] for r in results]
        except Exception:
            # 重排服务挂了/超时: 降级返回RRF融合结果, 保证检索链路不中断
            return docs[: self.k_final]

    # ---------- 对外主入口 ----------
    def retrieve(self, query: str):
        vector_docs = self._vector_recall(query)  # 语义路
        bm25_docs = self._bm25_recall(query)      # 关键词路
        fused = self._rrf_fuse([vector_docs, bm25_docs], k=self.k_fuse)
        return self._rerank(query, fused)

    def invoke(self, query: str):
        # LCEL链里可以像retriever一样被管道调用
        return self.retrieve(query)


# ---------- 全局单例: 上传新文档后由knowledge_base_service调用refresh刷新 ----------
_hybrid = None

def get_hybrid_retriever():
    global _hybrid
    if _hybrid is None:
        from vector_store_service import VectorStoreService
        from langchain_community.embeddings import DashScopeEmbeddings
        vector_store = VectorStoreService(embedding=DashScopeEmbeddings())
        _hybrid = HybridRetriever(vector_store.vector_store)
    return _hybrid
