#!/usr/bin/env python3
"""Create a browsable HTML report for the 50-entity image comparison."""

import argparse
import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_DIR = ROOT / "MMKGC/outputs/mkgw_image_comparison_50"
DEFAULT_DESCRIPTIONS = ROOT / "MMKGC/outputs/mkgw_image_test_50_20261014_fixed2/entity_descriptions.csv"


def load_descriptions(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["entity_key"]: row for row in csv.DictReader(handle)}


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def relative_image_path(report_dir, copied_path):
    path = Path(copied_path)
    return path.relative_to(report_dir).as_posix()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, default=DEFAULT_COMPARISON_DIR)
    parser.add_argument("--descriptions", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    comparison_dir = args.comparison_dir.resolve()
    output = (args.output or comparison_dir / "comparison_report.html").resolve()
    manifest = json.loads((comparison_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    descriptions = load_descriptions(args.descriptions)

    toc = []
    sections = []
    for entity in manifest["entities"]:
        key = entity["entity_key"]
        idx = entity["idx"]
        label = entity["label"]
        anchor = f"entity-{idx:03d}"
        row = descriptions.get(key, {})
        description = row.get("description", "")
        folder = Path(entity["original"]["copied"]).parent.name

        original = relative_image_path(comparison_dir, entity["original"]["copied"])
        mmkgc = relative_image_path(comparison_dir, entity["mmkgc"]["copied"])
        legacy = relative_image_path(comparison_dir, entity["legacy_knowledge_bridger"]["copied"])
        mmkgc_score = entity["mmkgc"]["score"]
        legacy_score = entity["legacy_knowledge_bridger"]["score"]

        toc.append(
            f'<li><a href="#{anchor}">{idx + 1}. {esc(label)}</a> '
            f'<span class="qid">{esc(key)}</span></li>'
        )
        sections.append(f"""
        <article class="entity" id="{anchor}">
          <div class="entity-header">
            <div>
              <h2>{idx + 1}. {esc(label)}</h2>
              <div class="meta">{esc(key)} · {esc(entity.get('category', row.get('category', '')))}</div>
            </div>
            <a class="back" href="#top">返回目录</a>
          </div>
          <div class="images">
            <figure>
              <img src="{esc(original)}" alt="{esc(label)} 原图" loading="lazy">
              <figcaption><strong>原图</strong></figcaption>
            </figure>
            <figure>
              <img src="{esc(mmkgc)}" alt="{esc(label)} MMKGC 最优生成图" loading="lazy">
              <figcaption><strong>MMKGC 最优图</strong><br>候选：{esc(entity['mmkgc']['candidate_id'])}<br>分数：{esc(mmkgc_score)}</figcaption>
            </figure>
            <figure>
              <img src="{esc(legacy)}" alt="{esc(label)} legacy Knowledge-Bridger 最优生成图" loading="lazy">
              <figcaption><strong>legacy Knowledge-Bridger 最优图</strong><br>候选：id_{esc(entity['legacy_knowledge_bridger']['candidate_index'])}.jpg<br>分数：{esc(legacy_score)}</figcaption>
            </figure>
          </div>
          <section class="description">
            <h3>文本描述</h3>
            <p>{esc(description)}</p>
          </section>
          <div class="sources">
            图片目录：<code>{esc(folder)}</code>
          </div>
        </article>
        """)

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MMKGC 与 legacy Knowledge-Bridger 图片对比</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18212b;
      --muted: #5f6b76;
      --line: #d8dee5;
      --panel: #f5f7f9;
      --accent: #1769aa;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; color: var(--ink); background: #fff; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.6; }}
    main {{ width: min(1400px, calc(100% - 40px)); margin: 0 auto; }}
    header {{ padding: 42px 0 26px; border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 8px; font-size: clamp(26px, 4vw, 42px); line-height: 1.15; }}
    h2 {{ margin: 0; font-size: 25px; line-height: 1.25; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    .subtitle, .meta, .sources, .qid {{ color: var(--muted); }}
    .subtitle {{ max-width: 900px; margin: 0; }}
    .notice {{ margin-top: 18px; padding: 12px 14px; border-left: 4px solid var(--accent); background: #eef6fd; }}
    .toc {{ padding: 24px 0 10px; border-bottom: 1px solid var(--line); }}
    .toc h2 {{ font-size: 20px; margin-bottom: 8px; }}
    .toc ol {{ columns: 3 260px; column-gap: 34px; margin: 0; padding-left: 24px; }}
    .toc li {{ break-inside: avoid; margin: 2px 0; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .qid {{ font-size: 12px; }}
    .entity {{ padding: 34px 0 42px; border-bottom: 1px solid var(--line); scroll-margin-top: 18px; }}
    .entity-header {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 18px; }}
    .back {{ white-space: nowrap; font-size: 14px; }}
    .images {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: var(--panel); border: 1px solid var(--line); padding: 10px; }}
    figure img {{ display: block; width: 100%; height: 320px; object-fit: contain; background: #e9edf1; }}
    figcaption {{ min-height: 70px; padding: 9px 2px 2px; color: var(--muted); font-size: 14px; }}
    figcaption strong {{ color: var(--ink); }}
    .description {{ margin-top: 20px; padding: 16px 18px; background: var(--panel); border: 1px solid var(--line); }}
    .description p {{ margin: 0; white-space: pre-wrap; }}
    .sources {{ margin-top: 10px; font-size: 12px; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    @media (max-width: 760px) {{
      main {{ width: min(100% - 24px, 680px); }}
      .images {{ grid-template-columns: 1fr; }}
      figure img {{ height: auto; max-height: 420px; }}
      .entity-header {{ display: block; }}
      .back {{ display: inline-block; margin-top: 8px; }}
    }}
    @media print {{
      main {{ width: 100%; }}
      header, .toc {{ break-after: page; }}
      .entity {{ break-before: page; }}
      .back {{ display: none; }}
      figure img {{ height: 250px; }}
      a {{ color: inherit; }}
    }}
  </style>
</head>
<body id="top">
  <main>
    <header>
      <h1>MMKGC 与 legacy Knowledge-Bridger 图片对比</h1>
      <p class="subtitle">50 个实体的原图、MMKGC 最高分生成图和 legacy Knowledge-Bridger 最高分生成图对照。</p>
      <div class="notice">注意：两种方法使用的评分模型和评分尺度不同。这里的“最高分”只表示各自方法内部排序后的最佳候选，不能直接比较两种方法的分数大小。</div>
    </header>
    <nav class="toc">
      <h2>实体目录（50 个）</h2>
      <ol>{''.join(toc)}</ol>
    </nav>
    {''.join(sections)}
  </main>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    print(f"report_entities={len(manifest['entities'])}")
    print(f"report={output}")


if __name__ == "__main__":
    main()
