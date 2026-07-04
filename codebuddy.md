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

## 7. 当前进度

- [x] 初始化 git 仓库、目录结构、规范
- [ ] 完成深度版调研报告（docs2/research--20260704--v1.md）
- [ ] 完成 design 方案设计文档
- [ ] 完成 plan 开发计划文档
