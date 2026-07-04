#!/usr/bin/env python3
"""LLM 批量打标：读取论文标题+摘要，选择分类标签。

用法：
    uv run python scripts/tag_papers.py

输入：
    data/paper_metadata.csv

产出：
    data/paper_metadata.csv（新增 tags 列）
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

# ─── 分类体系 ────────────────────────────────────────────

TAXONOMY = """
教育不平等 (Education Inequality)
  ├── 学校隔离 (School Segregation)
  ├── 居住隔离与学校 (Residential-School Link)
  ├── 教育市场化 (Education Marketization)
  ├── 文化资本 (Cultural Capital)
  ├── EMI/MMI 理论 (Effectively/Maximally Maintained Inequality)
  └── 教育政策 (Education Policy)

空间分析 (Spatial Analysis)
  ├── 空间可达性 (Spatial Accessibility)
  ├── 空间自相关 (Spatial Autocorrelation)
  ├── 城市服务设施 (Urban Services/Facilities)
  └── 绅士化 (Gentrification)

研究方法 (Methodology)
  ├── 量化方法 (Quantitative)
  ├── 混合方法 (Mixed Methods)
  └── 综述/理论 (Review/Theory)

地理区域 (Region)
  ├── 中国 (China)
  ├── 欧洲 (Europe)
  ├── 北美 (North America)
  ├── 南美 (South America)
  ├── 亚洲其他 (Other Asia)
  └── 全球/跨国 (Global/Cross-national)
"""

TAGGING_PROMPT = """You are a research librarian classifying academic papers. Based on the paper's title and abstract, select 1-3 most appropriate labels from the taxonomy below.

Taxonomy:
{taxonomy}

Paper Title: {title}
Abstract: {abstract}

Return ONLY a JSON array of labels (exact taxonomy paths), like:
["教育不平等/学校隔离", "中国"]

Labels:"""


def extract_abstract(pdf_path: str, max_chars: int = 1500) -> str:
    """从 PDF 提取摘要（前 2 页文本的前 1500 字符）。"""
    try:
        import fitz
    except ImportError:
        return ""

    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return ""

    text = ""
    for page_no in range(min(2, len(doc))):
        text += doc[page_no].get_text()
    doc.close()

    # 尝试找到 Abstract 段落
    abstract_match = re.search(
        r"(?:Abstract|ABSTRACT|摘要)[\s\n]+(.+?)(?:\n\n|\n(?:Introduction|1\.|Keywords|KEYWORDS|关键词))",
        text, re.DOTALL
    )
    if abstract_match:
        text = abstract_match.group(1).strip()

    return text[:max_chars]


def tag_papers(csv_path: str = "data/paper_metadata.csv", delay: float = 0.5):
    """批量打标。"""
    # 读 CSV
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"共 {len(rows)} 篇论文待打标")

    # 初始化 LLM
    sys.path.insert(0, "src")
    from georesearcher.models.llm import get_llm

    llm = get_llm()

    # 检查已有的标签
    already_tagged = sum(1 for r in rows if r.get("tags", "").strip())
    print(f"已有标签: {already_tagged}, 待打标: {len(rows) - already_tagged}")

    for i, row in enumerate(rows):
        if row.get("tags", "").strip():
            continue  # 已有标签，跳过

        title = row["title"]
        pdf_path = row["dest_path"]

        print(f"[{i+1}/{len(rows)}] {title[:60]}...", end=" ")

        # 提取摘要
        abstract = extract_abstract(pdf_path)
        if not abstract:
            abstract = title  # fallback

        # 调用 LLM
        prompt = TAGGING_PROMPT.format(
            taxonomy=TAXONOMY.strip(),
            title=title,
            abstract=abstract,
        )

        try:
            resp = llm.complete(prompt, temperature=0.1)
            # 提取 JSON 数组
            json_match = re.search(r"\[.*?\]", resp, re.DOTALL)
            if json_match:
                tags = json.loads(json_match.group(0))
                if isinstance(tags, list) and len(tags) > 0:
                    row["tags"] = "; ".join(tags)
                    print(f"-> {row['tags']}")
                else:
                    print("-> [NO TAGS]")
                    row["tags"] = ""
            else:
                print(f"-> [PARSE FAIL] {resp[:100]}")
                row["tags"] = ""
        except Exception as e:
            print(f"-> [ERROR] {e}")
            row["tags"] = ""

        # 逐行写回 CSV（支持中断续跑）
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        time.sleep(delay)

    tagged = sum(1 for r in rows if r.get("tags", "").strip())
    print(f"\n打标完成: {tagged}/{len(rows)}")


if __name__ == "__main__":
    tag_papers()
