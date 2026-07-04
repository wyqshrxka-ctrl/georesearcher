"""RAG 检索管道：混合检索（稠密+稀疏）→ RRF 融合 → 交叉编码器重排序。

检索流程：
  1. SQLite tag 过滤 → 候选 paper_ids
  2. HyDE 生成假设答案
  3. 多查询变体并行
  4. 稠密检索（bge-m3, Chroma）+ 稀疏检索（BM25, 内存索引）
  5. RRF 融合稠密+稀疏结果
  6. 交叉编码器精细重排序（bge-reranker-v2-m3）
     fallback: LLM judge 重排序
  7. 父块上下文扩展（通过 section_idx 查 SQLite）
  8. TTLCache 缓存
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from ...config import Config, load_config
from ...models.embedding import Embedder, get_embedder
from ...models.llm import LLMClient, get_llm, get_judge
from ...storage.sqlite_store import SqliteStore, get_sqlite_store
from ...storage.vector_store import VectorStore, get_vector_store
from ...types import Chunk, Retrieved

# ─── 从 query 中提取隐式标签 ───────────────────────────────

_TAG_EXTRACTION_PROMPT = """From the following research question, extract any implicit geographic regions or research topics as labels from the taxonomy below. Return ONLY a JSON array of matching taxonomy paths, or an empty array [].

Taxonomy:
教育不平等/学校隔离, 教育不平等/居住隔离与学校, 教育不平等/教育市场化, 教育不平等/文化资本, 教育不平等/EMI/MMI 理论, 教育不平等/教育政策
空间分析/空间可达性, 空间分析/空间自相关, 空间分析/城市服务设施, 空间分析/绅士化
研究方法/量化方法, 研究方法/混合方法, 研究方法/综述/理论
中国, 欧洲, 北美, 南美, 亚洲其他, 全球/跨国

Question: {question}

Labels:"""


# ─── HyDE prompt ────────────────────────────────────────────

_HYDE_PROMPT = """You are a helpful research assistant. Write a short paragraph (2-4 sentences) that answers the following research question, as if you were summarizing findings from a scientific paper. Do NOT cite any specific paper. Just give a plausible academic answer.

Question: {question}

Answer:"""


# ─── 多查询 prompt ───────────────────────────────────────────

_MULTI_QUERY_PROMPT = """Given the following research question, generate {num_queries} alternative versions that express the same information need from different angles. Use different keywords, synonyms, or rephrasings. Return ONLY the list, one per line, no numbering, no explanation.

Original question: {question}

Alternative versions:"""


# ─── 重排序 prompt (fallback: LLM judge) ──────────────────────

_RERANK_PROMPT = """You are a research relevance judge. Rate how relevant each passage is to the query on a scale of 1-10 (10=highly relevant). Return ONLY the scores as "passage_id: score", one per line.

Query: {query}

Passages:
{passages}

Scores:"""


# ─── 简易 TTLCache ──────────────────────────────────────────

class _TTLCache:
    def __init__(self, maxsize: int = 128, ttl_seconds: int = 300):
        self._maxsize = maxsize
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[float, list[Retrieved]]] = OrderedDict()

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, query: str) -> list[Retrieved] | None:
        key = self._key(query)
        if key in self._store:
            ts, val = self._store[key]
            if time.monotonic() - ts < self._ttl:
                self._store.move_to_end(key)
                return val
            del self._store[key]
        return None

    def set(self, query: str, results: list[Retrieved]) -> None:
        key = self._key(query)
        self._store[key] = (time.monotonic(), results)
        self._store.move_to_end(key)
        if len(self._store) > self._maxsize:
            self._store.popitem(last=False)


# ─── 分词工具 ──────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """中英混合分词：空格分割 + 中文单字 + 标点过滤。"""
    import re as _re
    # 先按空格和标点切分
    raw = _re.split(r"[，。！？；：""（）【】《》\s,\.!?;:()\[\]{}]+", text)
    tokens = []
    for token in raw:
        if not token:
            continue
        # 中文部分拆成单字，英文部分保留原词
        sub = _re.split(r"([\u4e00-\u9fff])", token)
        for s in sub:
            s = s.strip().lower()
            if s and len(s) >= 2 or (_re.match(r"^[\u4e00-\u9fff]$", s)):
                tokens.append(s)
    return tokens


# ─── BM25 稀疏检索索引 ─────────────────────────────────────

class _BM25Index:
    """内存 BM25 索引：懒加载 Chroma 全部文档构建。

    设计：首次 retrieve() 时从 Chroma 拉取所有文档 → 分词 → 构建 BM25Okapi。
    25k 文档量级内存可行（约 50-100 MB）。
    """

    def __init__(self, vector_store):
        self._vs = vector_store
        self._bm25 = None
        self._doc_ids: list[str] = []
        self._paper_ids: list[str] = []
        self._doc_texts: list[str] = []

    def _ensure(self):
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        col = self._vs._ensure()
        all_data = col.get(include=["documents", "metadatas"])
        self._doc_ids = all_data["ids"]
        self._doc_texts = all_data["documents"] or []
        self._paper_ids = [
            (m or {}).get("paper_id", "") for m in (all_data.get("metadatas") or [])
        ]
        tokenized = [_tokenize(d) for d in self._doc_texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(
        self, query: str, top_k: int, filters: dict | None = None
    ) -> list[Retrieved]:
        """BM25 检索，可选 paper_id 过滤。"""
        self._ensure()
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # 构建结果列表
        results: list[Retrieved] = []
        for idx, score in enumerate(scores):
            pid = self._paper_ids[idx] if idx < len(self._paper_ids) else ""

            # 应用 paper_id 过滤
            if filters and "$in" in filters:
                allowed = set(filters["$in"])
                if pid not in allowed:
                    continue

            results.append(
                Retrieved(
                    chunk=Chunk(
                        id=self._doc_ids[idx],
                        paper_id=pid,
                        text=self._doc_texts[idx],
                        level="para",
                    ),
                    score=float(score),
                )
            )

        # 按分数降序排列
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ─── 交叉编码器重排序 ──────────────────────────────────────

class _CrossEncoderReranker:
    """本地交叉编码器重排序（bge-reranker-v2-m3）。

    懒加载：首次 rerank() 时加载模型，遵循 Embedder 延迟加载模式。
    """

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        from sentence_transformers.cross_encoder import CrossEncoder

        self._model = CrossEncoder("BAAI/bge-reranker-v2-m3", device=self._device)

    def rerank(
        self, query: str, candidates: list[Retrieved], top_k: int
    ) -> list[Retrieved]:
        """交叉编码器精细排序。"""
        self._ensure()
        if not candidates:
            return candidates

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)

        # 将得分附加到结果
        indexed = list(zip(candidates, scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in indexed[:top_k]]


# ─── RRF 融合 ───────────────────────────────────────────────

def _rrf_fusion(
    dense_results: list[Retrieved],
    sparse_results: list[Retrieved],
    k: int = 60,
) -> list[Retrieved]:
    """Reciprocal Rank Fusion：融合稠密+稀疏检索结果。

    RRF 公式：score = sum(1 / (k + rank_i)) for each retriever i

    优点：不需要调权重，对排名差异鲁棒。
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, Retrieved] = {}

    for rank, r in enumerate(dense_results):
        rrf_scores[r.chunk.id] = rrf_scores.get(r.chunk.id, 0) + 1.0 / (k + rank + 1)
        chunk_map[r.chunk.id] = r

    for rank, r in enumerate(sparse_results):
        rrf_scores[r.chunk.id] = rrf_scores.get(r.chunk.id, 0) + 1.0 / (k + rank + 1)
        if r.chunk.id not in chunk_map:
            chunk_map[r.chunk.id] = r

    # 按 RRF 分数降序排列
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    return [
        Retrieved(chunk=chunk_map[cid].chunk, score=rrf_scores[cid])
        for cid in sorted_ids
    ]


# ─── RetrievalPipeline ──────────────────────────────────────

@dataclass
class RetrievalConfig:
    """检索配置（运行时，可被 CLI 覆盖）。"""
    top_k: int = 5
    use_hyde: bool = True
    use_multi_query: bool = True
    use_reranker: bool = True
    use_tag_filter: bool = True
    expand_parent: bool = True
    use_bm25: bool = True
    use_cross_encoder: bool = True
    rrf_k: int = 60
    hybrid_candidates: int = 40     # 各检索器召回数
    rerank_candidates: int = 20     # 进入交叉编码器的候选数
    multi_query_count: int = 3
    hyde_temperature: float = 0.7


@dataclass
class RetrievalResult:
    """增强版检索结果：含子块 + 父块上下文。"""
    child_chunk: Retrieved
    parent_text: str = ""  # section 全文


class RetrievalPipeline:
    """RAG 检索管道：混合检索 + 交叉编码器重排序。"""

    def __init__(
        self,
        cfg: Config | None = None,
        llm: LLMClient | None = None,
        judge: LLMClient | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        sqlite_store: SqliteStore | None = None,
    ):
        cfg = cfg or load_config()
        self._llm = llm or get_llm(cfg)
        self._judge = judge or get_judge(cfg)
        self._embedder = embedder or get_embedder(cfg)
        self._vector_store = vector_store or get_vector_store(cfg)
        self._sqlite_store = sqlite_store or get_sqlite_store(cfg)
        self._retrieval_cfg = cfg.retrieval
        self._embedding_device = cfg.models.embedding.device
        self._cache = _TTLCache(maxsize=64, ttl_seconds=600)

        # 懒加载：首次使用时才初始化
        self._bm25_index: _BM25Index | None = None
        self._cross_encoder: _CrossEncoderReranker | None = None

    def retrieve(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
        skip_cache: bool = False,
    ) -> list[RetrievalResult]:
        """主检索入口：混合检索 → RRF 融合 → 交叉编码器重排序 → 父块扩展。"""
        final = self.retrieve_raw(query, config=config, skip_cache=skip_cache)
        return self._results_to_retrieval_results(final)

    def retrieve_raw(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
        skip_cache: bool = False,
    ) -> list[Retrieved]:
        """返回排序后的原始命中（父块扩展前），供评估层计算检索指标用。

        与 retrieve() 的区别：不做父块扩展，直接返回 list[Retrieved]，
        便于取每个命中的 chunk.paper_id 与排名（Hit@k / MRR / NDCG）。
        """
        rc = config or self._build_config()

        use_cache = not skip_cache and len(query) > 20
        if use_cache:
            cached = self._cache.get(query)
            if cached is not None:
                return cached[: rc.top_k]

        final = self._retrieve_ranked(query, rc)

        if use_cache:
            self._cache.set(query, final)

        return final

    def _retrieve_ranked(
        self, query: str, rc: RetrievalConfig
    ) -> list[Retrieved]:
        """核心检索逻辑：tag 过滤 → 多查询 → 稠密+稀疏 → RRF → 重排序 → top_k。"""
        # 1. 标签预过滤
        candidate_ids: list[str] | None = None
        if rc.use_tag_filter:
            implicit_tags = self._extract_tags_from_query(query)
            if implicit_tags:
                candidate_ids = self._sqlite_store.search_by_tags(implicit_tags)
                if not candidate_ids:
                    return []

        # 2. 多查询变体
        if rc.use_multi_query:
            queries = self._generate_multi_queries(query, rc.multi_query_count)
        else:
            queries = [query]

        # 3. 对每个查询进行稠密 + 稀疏检索
        all_dense: dict[str, Retrieved] = {}
        all_sparse: dict[str, Retrieved] = {}

        for q in queries:
            search_text = q
            if rc.use_hyde:
                hyde_answer = self._generate_hyde(q)
                search_text = f"{q} {hyde_answer}"

            # 构建过滤条件
            chroma_filter = None
            if candidate_ids:
                chroma_filter = {"paper_id": {"$in": candidate_ids}}

            # 3a. 稠密检索
            query_emb = self._embedder.embed_one(search_text)
            dense_results = self._vector_store.search(
                query_emb, top_k=rc.hybrid_candidates, filters=chroma_filter
            )
            for r in dense_results:
                cid = r.chunk.id
                if cid not in all_dense or r.score > all_dense[cid].score:
                    all_dense[cid] = r

            # 3b. 稀疏检索 (BM25)
            if rc.use_bm25:
                if self._bm25_index is None:
                    self._bm25_index = _BM25Index(self._vector_store)
                sparse_results = self._bm25_index.search(
                    q, top_k=rc.hybrid_candidates, filters=chroma_filter
                )
                for r in sparse_results:
                    cid = r.chunk.id
                    if cid not in all_sparse or r.score > all_sparse[cid].score:
                        all_sparse[cid] = r

        # 4. RRF 融合（如果有 BM25 结果）
        dense_list = list(all_dense.values())
        sparse_list = list(all_sparse.values())

        if rc.use_bm25 and sparse_list:
            candidates = _rrf_fusion(dense_list, sparse_list, k=rc.rrf_k)
        else:
            # 纯稠密：直接排序
            dense_list.sort(key=lambda x: x.score, reverse=True)
            candidates = dense_list

        # 5. 交叉编码器重排序（优先）或 LLM judge 重排序（fallback）
        if len(candidates) > rc.top_k:
            if rc.use_cross_encoder:
                if self._cross_encoder is None:
                    self._cross_encoder = _CrossEncoderReranker(
                        device=self._embedding_device
                    )
                top_for_rerank = candidates[: rc.rerank_candidates]
                candidates = self._cross_encoder.rerank(query, top_for_rerank, rc.top_k)
            elif rc.use_reranker:
                candidates = self._rerank(query, candidates, rc.top_k)

        return candidates[: rc.top_k]

    def _results_to_retrieval_results(
        self, retrieved: list[Retrieved]
    ) -> list[RetrievalResult]:
        """为每个子块加载父块上下文。"""
        out: list[RetrievalResult] = []
        seen: set[tuple[str, int]] = set()

        for r in retrieved:
            chunk = r.chunk
            sidx = 0
            match = re.search(r"_s(\d+)_p", chunk.id)
            if match:
                sidx = int(match.group(1))

            parent_text = ""
            cache_key = (chunk.paper_id, sidx)
            if cache_key not in seen:
                parent_text = self._sqlite_store.get_parent_chunk(chunk.paper_id, sidx) or ""
                seen.add(cache_key)

            out.append(RetrievalResult(
                child_chunk=r,
                parent_text=parent_text,
            ))

        return out

    def _extract_tags_from_query(self, query: str) -> list[str]:
        """从问题中提取隐式标签（中国、学校隔离等）。"""
        quick_tags = []
        region_map = {
            "中国": "中国", "深圳": "中国", "北京": "中国", "上海": "中国",
            "欧洲": "欧洲", "英国": "欧洲", "法国": "欧洲", "瑞典": "欧洲", "荷兰": "欧洲",
            "美国": "北美", "加拿大": "北美",
            "巴西": "南美", "智利": "南美", "墨西哥": "南美",
            "印度": "亚洲其他", "日本": "亚洲其他", "韩国": "亚洲其他",
        }
        topic_map = {
            "学校隔离": "教育不平等/学校隔离",
            "隔离": "教育不平等/学校隔离",
            "居住隔离": "教育不平等/居住隔离与学校",
            "市场化": "教育不平等/教育市场化",
            "文化资本": "教育不平等/文化资本",
            "教育政策": "教育不平等/教育政策",
            "空间可达": "空间分析/空间可达性",
            "可达性": "空间分析/空间可达性",
            "服务设施": "空间分析/城市服务设施",
            "绅士化": "空间分析/绅士化",
            "空间自相关": "空间分析/空间自相关",
            "教育不平等": "教育不平等",
            "通勤": "空间分析/空间可达性",
            "学区": "教育不平等/教育政策",
        }

        for kw, tag in {**region_map, **topic_map}.items():
            if kw in query and tag not in quick_tags:
                quick_tags.append(tag)

        return quick_tags[:3]

    def _build_config(self) -> RetrievalConfig:
        rc = self._retrieval_cfg
        return RetrievalConfig(
            top_k=rc.top_k,
            use_hyde=rc.use_hyde,
            use_multi_query=rc.use_multi_query,
            use_reranker=rc.use_reranker,
            use_bm25=rc.use_bm25,
            use_cross_encoder=rc.use_cross_encoder,
            rrf_k=rc.rrf_k,
            hybrid_candidates=rc.hybrid_candidates,
            rerank_candidates=rc.rerank_candidates,
        )

    def _generate_hyde(self, query: str) -> str:
        prompt = _HYDE_PROMPT.format(question=query)
        try:
            return self._llm.complete(prompt, temperature=0.7)
        except Exception:
            # HyDE 失败则退回纯 query 检索，不让整条检索崩
            return ""

    def _generate_multi_queries(self, query: str, count: int) -> list[str]:
        prompt = _MULTI_QUERY_PROMPT.format(question=query, num_queries=count)
        try:
            resp = self._llm.complete(prompt, temperature=0.7)
        except Exception:
            # 多查询扩展失败则退回单查询
            return [query]
        lines = [
            line.strip().lstrip("-•0123456789. ")
            for line in resp.split("\n")
            if line.strip()
        ]
        return [query] + [q for q in lines if q][:count - 1]

    def _rerank(
        self, query: str, candidates: list[Retrieved], top_k: int
    ) -> list[Retrieved]:
        """LLM judge 重排序（fallback，当交叉编码器禁用时使用）。"""
        passages = "\n\n".join(
            f"[passage_{i}] {r.chunk.text[:500]}" for i, r in enumerate(candidates)
        )
        prompt = _RERANK_PROMPT.format(query=query, passages=passages)
        resp = self._judge.complete(prompt, temperature=0.0)

        scores: dict[int, float] = {}
        for line in resp.split("\n"):
            line = line.strip()
            parts = line.replace(":", " ").split()
            for i, part in enumerate(parts):
                if part.startswith("passage_"):
                    try:
                        pid = int(part.replace("passage_", "").rstrip(":"))
                        if i + 1 < len(parts):
                            scores[pid] = float(parts[i + 1])
                    except (ValueError, IndexError):
                        continue

        if not scores:
            return candidates[:top_k]

        indexed = list(enumerate(candidates))
        indexed.sort(key=lambda x: scores.get(x[0], 0), reverse=True)
        return [c for _, c in indexed[:top_k]]
