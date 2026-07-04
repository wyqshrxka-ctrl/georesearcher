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
├── .gitignore            # docs2/ 不入 git
├── docs/                 # 开发文档（plan/design 等），入 git
└── docs2/                # 调研报告等，不入 git
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
- [x] 完成面试追问应答手册（docs/interview--20260704--v1.md）
- [x] **M0 骨架完成**：pyproject（Python 3.11, uv）、config.yaml+.env.example、四层分层、types.py（pydantic 共享结构）、模型层（DeepSeek LLM/judge + 本地 bge-m3，均延迟加载）、存储层（VectorStore 协议 + Chroma 实现 + Milvus 桩 + SQLite papers/notes/citations）、CLI（version/doctor）、README、8 个单测全绿。commit 8da7757。
- [ ] M1 RAG 内核重构（下一步）

## 9. 关键工程事实（供后续会话）
- Python 3.11（/opt/homebrew，uv 管理）；venv 在 .venv/。运行命令用 `uv run georesearcher ...` 或 `uv run pytest`。
- 重依赖（llama-index/sentence-transformers/pymupdf/ragas/streamlit/mcp）是 pyproject 的 optional extras（rag/eval/ui/gis），M0 未装，按里程碑再装。
- 模型/embedding 均延迟加载：无 API key、未装 rag extra 时骨架仍可跑（doctor/smoke 不触发网络与模型下载）。
- data/、.venv/、.env 均已 gitignore；uv.lock 入库保证可复现。
