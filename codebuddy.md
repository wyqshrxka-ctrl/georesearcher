# research_agent — 项目记忆（codebuddy.md）

> 本文件是项目的长期记忆，入 git 做版本化管理。每次重要决策、架构调整、规范变更都应更新此文件。

---

## 1. 项目定位

- **一句话**：面向 GIS / 时空大数据 / 智慧城市领域的**科研 Agent**，串起「文献检索 → 知识库入库 → 文献解读 → 创新点发现 → 代码实验 → 绘图 → 学术写作」的科研闭环。
- **首要目的**：**简历项目**。用于应聘「大模型应用开发 / Agent 开发」岗位。因此评判标准是 **架构讲得清、技术选型有 trade-off、有可展示 demo、有工程亮点**，而非功能大而全。
- **次要目的**：作者本人日常科研自用工具。

## 2. 作者背景与约束

- 领域：GIS、时空大数据、智慧城市，部分社会科学结合。常用 geopandas、POI、专题地图、统计图表；有 QGIS 工作流（QGIS 可通过 MCP 操作）。目标期刊如 *Nature Cities*。
- 技术：主力 Python。**基本不手写代码，全部由 AI 生成**。因此代码要清晰、可维护、注释充分、模块边界明确。
- 模型：优先接入 **DeepSeek**（成本考虑）。架构上模型层要可切换（预留 OpenAI/Claude 兼容接口）。
- 展示形态：需要能放 GitHub 展示、简历好看；倾向 **CLI 内核 + Web UI（Streamlit/Gradio）demo**。

## 3. 已有资产：需融合的旧项目「研问」

- 时间：2026.01–2026.03，用 Claude Code 写的 demo，有不成熟处，本项目要**重构并融合**为新 Agent 的知识库内核。
- 旧技术栈：Python、FastAPI、LlamaIndex、Milvus、MySQL、Vue 3。
- 旧亮点：HyDE + 多查询并行 + Reranker 三阶段检索；TTLCache 缓存 + 短查询直通；PyMuPDF 结构化解析；LlamaIndex 三级粒度分块；Milvus HNSW；本地 Embedding；流式响应；APA 引用自动生成。
- **重构重点（面试痛点）**：作者面试被追问「**如何评测 RAG 效果**」。新项目必须内建 **RAG 评估模块**（检索层 Hit Rate/MRR/NDCG/Context Precision-Recall；生成层 Faithfulness/Answer Relevancy；框架 RAGAS/TruLens；评估→定位→优化 闭环）。

## 4. 关键决策 / 红线

- **文献来源走合法路径**：arXiv API、Semantic Scholar API、OpenAlex API、Unpaywall（找开放获取版本）+ 本地 PDF 手动投喂。**禁止**集成 Sci-Hub 等盗版源到代码/简历（合规是面试加分项，反之是硬伤）。
- 「自动跑实验闭环」在 GIS+社科领域先降级为「**生成分析代码 + 沙箱执行 + 看结果**」的半闭环，不做 AI-Scientist 式全自动。
- 绘图模块聚焦：统计图表 + 专题地图（choropleth 等），参考顶刊风格模板；不涉及生物/化学类图。
- GIS/QGIS-MCP 是本项目相对通用科研 agent 的**差异化亮点**，重点打造。

## 5. 目录结构

```
research_agent/
├── codebuddy.md          # 本文件，项目记忆，入 git
├── config.yaml           # 集中配置（models/storage/retrieval/evaluation）
├── docs/                 # 开发文档（plan/design），入 git
├── docs2/                # 调研报告/面试手册，不入 git
├── scripts/              # batch_ingest / gen_eval_set 等运维脚本
├── src/georesearcher/
│   ├── capabilities/     # 能力层：ingest / rag / evaluation / gis / plotting …
│   ├── storage/          # 存储层：VectorStore(Chroma/Milvus) + SqliteStore
│   ├── models/           # 模型层：LLM(DeepSeek) / judge / 本地 embedder
│   └── ui/cli.py         # CLI 内核（version/doctor/ingest/ask/eval/eval-diff）
├── tests/                # 单测 + tests/eval_set/（评估集，入 git）
└── data/                 # chroma / sqlite / pdfs / reports（gitignore）
```

## 6. 文档命名规范

- 格式：`{类型}--{YYYYMMDD}--{版本号}.md`，例：`research--20260704--v1.md`、`design--20260705--v1.md`。
- 类型定义：
  - **plan** — 开发的详细计划（里程碑、任务拆解、排期）。
  - **design** — 方案设计文档（架构、模块、技术选型、接口）。
  - **research** — 调研报告（竞品/技术调研）。
- 版本号从 `v1` 起，同一文档重大修订递增（`v2`、`v3`…）。
- 放置：`plan`/`design` → `docs/`（入 git）；`research` → `docs2/`（不入 git）。

## 7. 架构决策（2026-07-04 grill 结论，10 条）

1. **做深模块**：(a)RAG+评估、(f)GIS工具链、(g)顶刊绘图 三个做深；(b)检索入库、(c)解读归档 做 demo 级；(d)找创新点、(h)学术写作 初期预留占位、后续接成熟 skill（如 awesome-ai-research-writing）；**(e)跑实验闭环先不做，仅预留接口**。
2. **multi-agent 形态**：**编排式（orchestrated）**，非真自主 multi-agent（避免 agent 间通信/失控/成本翻倍）。目标编排式，路径可先单 agent 跑通再拆专职节点。
3. **编排框架**：**LangGraph** 管编排 + **LlamaIndex** 管 RAG/知识库，**编排层与检索层解耦**。
4. **向量库**：抽象 `VectorStore` **双模式**——默认 **Chroma**（嵌入式、demo 友好），可切 **Milvus**（生产/简历资产）。
5. **结构化存储**：**SQLite**（文献约 200 篇够用）；引用关系存 citation edges 表，**预留 Neo4j 知识图谱升级钩子**。
6. **Embedding**：**本地 bge-m3**（中英跨语言 + 稠密/稀疏混合检索）+ bge-small 快速档；不做抽象（避免过度设计）。
7. **评估 judge model**：**可配置，默认 DeepSeek**，验证期可切强模型交叉对照 + 人工抽检校准。需掌握 LLM-as-judge 三坑（自我偏好/位置长度偏差/一致性）。
8. **可维护性=硬约束**：四层分层（编排层/能力层/存储层/模型层，层间走接口）；每个能力模块=独立 package + 可单测；配置集中；AI 改动须附测试。
9. **交付形态**：**CLI/Python 内核 + Streamlit demo**（逻辑与 UI 解耦）；README 做得有展示力（架构图/徽章/gif）。不用 Vue3（留在旧项目简历）。
10. **文档**：写 **ADR（架构决策记录）** 进 README + docs；额外产出**面试追问应答手册**。

> 决策依据的原则：可展示性优先、可维护 > 花哨、stability > novelty；该抽象的抽象（VectorStore/judge），不该抽的不抽（embedding）。

## 8. 当前进度

- [x] 初始化 git 仓库、目录结构、规范
- [x] 完成深度版调研报告（docs2/research--20260704--v1.md）
- [x] 完成架构 grill（10 条决策）
- [x] 完成 design 方案设计文档（docs/design--20260704--v1.md，含执行者交接规范 §12、AI 开发方法论 §13）
- [x] 完成 plan 开发计划文档（docs/plan--20260704--v1.md）
- [x] 完成面试追问应答手册（docs2/interview--20260704--v1.md，**不入 git、不公开**）
- [x] **M0 骨架完成**：pyproject（Python 3.11, uv）、config.yaml+.env.example、四层分层、types.py（pydantic 共享结构）、模型层（DeepSeek LLM/judge + 本地 bge-m3，均延迟加载）、存储层（VectorStore 协议 + Chroma 实现 + Milvus 桩 + SQLite papers/notes/citations）、CLI（version/doctor）、README、8 个单测全绿。commit 8da7757。
- [x] **M1 RAG 内核完成**：HyDE + 多查询并行 + 混合检索（稠密 bge-m3 + 稀疏 BM25）+ RRF 融合 + 交叉编码器重排序（bge-reranker-v2-m3）+ TTLCache 缓存 + 父块扩展上下文 + APA 引用接地生成。知识库已导入 93 篇论文、25,827 个段落向量、629 个 section 父块。CLI 可完整问答。
- [x] **M2 RAG 评估完成**（2026-07-05）：分检索层 + 生成层两层评估，可跑通、有报告、有诊断、可回归对比。
  - 计划文档 `docs/plan-m2--20260705--v1.md`（为弱模型交接写的完整任务拆解 T0-T13）。
  - M1 前置修复：`retrieve_raw()` 返回原始 `list[Retrieved]` 供评估；HyDE/多查询容错；`_rerank` 补单测。
  - 评估模块 `capabilities/evaluation/`：schemas（pydantic）、dataset（加载校验）、retrieval_metrics（纯数学）、generation_metrics（自研 LLM-as-judge）、evaluator（编排 + 诊断）、report（rich 终端 + md + json）。
  - 造集脚本 `scripts/gen_eval_set.py`（single/group/mixed 三模式，按主题 tag 分组造跨篇多-GT 难题，半自动 + 人工校验）。
  - 评估集 `tests/eval_set/eval_set.json`（36 条，单篇 + 跨篇多-GT 混合，入 git）。
  - CLI 命令 `eval` + `eval-diff`；config 加 `evaluation` 段（含诊断阈值）。
  - **首次评估基线**（36 条，top_k=5）：检索层 Hit@5≈0.53 / MRR=0.50 / NDCG=0.44；诊断"检索层偏弱"。
  - **踩坑修复（写进面试文档）**：faithfulness 曾误得 0.107——按句号切句把 APA 参考文献列表切成碎片全判 no（76% 判定句是垃圾）；修复=`claim_sentences()` 过滤参考文献/话术/引用碎片 + 给 judge 完整上下文。
- [x] **M3 检索入库 + 解读归档完成**（2026-07-05）：OpenAlex 单源 + PaperSource 接口 + DOI 优先去重 + 摘要旁路入库 + interpret 结构化笔记；CLI `search`/`ingest-search`/`interpret`；独立库 `config.m3.yaml`→`./data/m3/`。真机端到端因 Cloudflare TLS fingerprinting 未通（curl 可通、Python urllib 被拦），已记录排障链于 `plan-m3--20260705--v1.md` §11。
- [x] **M4 编排设计完成**（2026-07-05）：经 grill 定 M4=LangGraph 编排（原 plan M4=GIS 顺延）。**三层 Agent 范式**（总控 Router/Intent-routing + RAG 质量控制 Corrective RAG + GIS 预留 ReAct）；完整 5 分支意图路由骨架，真接 SEARCH+ASK、PLOT/GIS/WRITE 占位；HumanReview 用 interrupt+SqliteSaver 断点续跑（b+）。计划文档 `plan-m4--20260705--v1.md`（T0-T9 可交接弱模型），ADR-16~19 入 design。langgraph 作 optional extra。
- [x] **M4 开发完成**（2026-07-05）：T0-T9 全部实现，211/211 测试全绿（新增 30 个 orchestration + 2 个 config 测试，零回归）。`orchestration/` 包（state/router/nodes/graph）、config（OrchestrationCfg + config.m4.yaml）、CLI（orchestrate 命令 3 模式）。图结构：1 动态路由 + 3 循环（重搜/CRAG/人机重答）+ 1 中断（interrupt+SqliteSaver 持久化）。langgraph==1.2.7 + checkpoint-sqlite==3.1.0。
- [ ] M2 收尾：T7 RAGAS 对照（可选）、design 回并 ADR-12~15
- [ ] 后续：GIS 工具链（MCP）/ 顶刊绘图 / Streamlit（编排之后独立里程碑）

## 9. 关键工程事实（供后续会话）
- Python 3.11（/opt/homebrew，uv 管理）；venv 在 .venv/。运行命令用 `uv run georesearcher ...` 或 `uv run pytest`。
- 重依赖（llama-index/sentence-transformers/pymupdf/ragas/streamlit/mcp）是 pyproject 的 optional extras（rag/eval/ui/gis），M0 未装，按里程碑再装。
- 模型/embedding 均延迟加载：无 API key、未装 rag extra 时骨架仍可跑（doctor/smoke 不触发网络与模型下载）。
- data/、.venv/、.env 均已 gitignore；uv.lock 入库保证可复现。
- **M1 混合检索**：稠密 bge-m3（Chroma）+ 稀疏 BM25（rank_bm25，内存索引 25k 文档，中英混合分词）+ RRF 融合（k=60）+ 交叉编码器 bge-reranker-v2-m3 重排序。配置项在 `config.yaml` 的 `retrieval` 段。
- **知识库规模**：93 篇论文，25,827 个段落向量（Chroma），629 个 section 父块（SQLite）。主题：教育不平等/学校隔离，中欧美跨地区。
- **向量化工具**：`scripts/batch_ingest.py` 支持两阶段导入（fast/vectorize），有断点续跑（`data/vectorized_papers.txt`）、tqdm 进度条、`status` 命令。
- **论文元数据**：`data/paper_metadata.csv`，含 93 篇论文的标题/作者/DOI/年份/LLM 标注标签。
- **M2 评估架构**：ground truth 是 **paper 级、二元相关性**（一条题标注哪几篇论文能答；chunk id 会随重向量化变，故用 paper 级）。检索指标 Hit@k/MRR/NDCG/Context Precision/Recall 全为纯数学（`retrieval_metrics.py`），有手算断言单测。生成指标 faithfulness/answer_relevancy 走自研 LLM-as-judge（`get_judge()`，temperature=0，用 yes/no 二元判断而非 1-5 打分）。
- **M2 评估集设计**：单篇 GT 会让 Recall/Precision 退化（|gold|=1 时 Recall 等价 Hit Rate、Precision 恒为 1/k），所以造了**多篇 GT 的跨篇题**（需综合 2-4 篇才能答），让这两个指标有分辨力。造集只用主题 tag（`教育不平等/*`、`空间分析/*`、`研究方法/*`）分组，排除纯地理 tag。
- **M2 测试策略**：评估器/指标测试全 mock（`RetrievalPipeline.__new__()` 注入 fake、fake judge），不真调 DeepSeek/不下模型，秒级跑完。真机端到端评估用 `uv run georesearcher eval`（慢：36 条各要检索 + 生成 + 逐句判 faithfulness，含生成层约 20-30 分钟）。
- **M2 已知教训**：LLM-as-judge 的 faithfulness，"怎么定义可判定的断言"比"用哪个 judge"更关键——机械按标点切句会被答案格式（参考文献/引用/Markdown）污染。遇到反常指标先怀疑测量工具、看分布判性质、拉最异常样本定位根因、量化验证、固化回归测试。
- **报告输出**：`data/reports/eval--{timestamp}.md|json`（已 gitignore）；`eval-diff run_a.json run_b.json` 做两次跑的指标 Δ 对比（回归防跷跷板）。
- **测试运行**：评估相关测试需 `uv run --extra dev --extra rag pytest tests/test_eval*.py tests/test_retrieval_metrics.py tests/test_generation_metrics.py tests/test_evaluator.py tests/test_report.py tests/test_gen_eval_set.py`。
- **M4 编排架构**：LangGraph StateGraph，共享 ResearchState（TypedDict + Annotated[list, operator.add] reducer），11 个节点（router/clarify/search/reflect/interpret/rag_qa/plot/gis/write/human_review），4 条条件边。Router 用 LLM 结构化 JSON 输出 + 置信度门控 + CLARIFY 节点反问。CRAG 用检索 top1 rerank 分（非 LLM 自评）触发联网补充。HumanReview 用标准 `interrupt()` + SqliteSaver 持久化（b+）。配置在 `orchestration` 段（7 个阈值）。`orchestrate` 命令 3 种模式：默认流畅（单命令内 interrupt→input→resume）、`--pause` 打印 thread_id 退出、`--resume <id> --feedback` 跨进程恢复。
- **M4 测试策略**：全 mock（零 LLM/零网络/零模型下载）。纯函数打 `_parse_intent_json`（7 个）+ `_route_after_*`（13 个条件边）。节点用 fake deps（FakeRetrievalPipeline/FakeGenerator/FakeLLM）。图集成用 fake 节点 + MemorySaver 验证走向/循环上限/中断恢复。
- **M4 运行**：需 `uv sync --extra orchestration --extra rag`；`uv run georesearcher --config config.m4.yaml orchestrate "..."`。M4 用独立库 `./data/m4/`（config.m4.yaml）。
