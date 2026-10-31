"""Render the English report from Markdown to a <= 8-page PDF.
把英文报告从 Markdown 渲染成不超过 8 页的 PDF。

The assignment requires a PDF report of at most 8 pages, so the page count is
checked here and the build fails loudly if the limit is exceeded.
作业要求报告 PDF 不超过 8 页，因此这里会检查页数，超限直接报错。

Usage / 用法:
    .venv/bin/python scripts/build_report_pdf.py [--md report/report.md] [--out report/report.pdf]
"""
from __future__ import annotations

import argparse
import os
import sys

import markdown
from pypdf import PdfReader
from xhtml2pdf import pisa

PAGE_LIMIT = 8

# Fonts: xhtml2pdf's built-in Helvetica lacks Greek/math glyphs (σ, ≈, −, ·) used in
# the report, so a system TTF is registered explicitly. Fallback list keeps the build
# working on machines without Arial.
# 字体：xhtml2pdf 内置 Helvetica 缺少报告用到的希腊字母/数学符号，显式注册系统 TTF；
# 找不到时回退，保证换机器也能构建。
FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
FONT_PATH = next((f for f in FONT_CANDIDATES if os.path.exists(f)), None)
FONT_FACE = (f"@font-face {{ font-family: ReportSans; src: url(file://{FONT_PATH}); }}"
             if FONT_PATH else "")

# Print CSS: A4, 10pt, tight margins so ~3k words of prose + tables fit in 8 pages.
# 打印样式：A4、10pt、窄边距，使 ~3k 词正文加表格能装进 8 页。
CSS = """
@page { size: A4; margin: 16mm 15mm 14mm 15mm; }
body { font-family: ReportSans, Helvetica, Arial, sans-serif; font-size: 9.4pt; line-height: 1.32; color: #111; }
h1 { font-size: 15pt; margin: 0 0 4pt 0; line-height: 1.2; }
h2 { font-size: 11.5pt; margin: 11pt 0 3pt 0; border-bottom: 0.6pt solid #bbb; padding-bottom: 1pt; }
h3 { font-size: 10pt; margin: 8pt 0 2pt 0; }
p { margin: 0 0 4pt 0; text-align: justify; }
ul, ol { margin: 0 0 4pt 14pt; padding: 0; }
li { margin: 0 0 1.5pt 0; }
code, pre { font-family: Courier, monospace; font-size: 8pt; }
pre { background: #f4f4f4; padding: 3pt; border-left: 2pt solid #ccc; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 6pt 0; font-size: 8.2pt; }
th, td { border: 0.5pt solid #999; padding: 2pt 3pt; vertical-align: top; }
th { background: #eee; }
blockquote { margin: 3pt 0 3pt 10pt; padding-left: 6pt; border-left: 2pt solid #ccc; color: #333; }
hr { border: 0; border-top: 0.5pt solid #ccc; margin: 6pt 0; }
a { color: #14417a; text-decoration: none; }
"""


# xhtml2pdf's font handling is unreliable for math glyphs even with an embedded TTF,
# so the PDF build transliterates them to ASCII. The Markdown source keeps the
# proper typography; only the rendered PDF is degraded, and deterministically so.
# xhtml2pdf 对数学符号的字体回退不可靠，故 PDF 构建时把符号转成 ASCII；
# Markdown 源保留原排版，只有渲染产物做确定性降级。
ASCII_MAP = {
    "≈": "~", "σ": "sigma", "α": "alpha", "λ": "lambda", "β": "beta",
    "−": "-", "–": "-", "—": "-", "×": "x", "·": ".", "μ": "mu",
    "≤": "<=", "≥": ">=", "→": "->", "Δ": "delta", "±": "+/-",
    "✔": "yes", "✘": "no", "✓": "yes", "✗": "no", "✱": "*",
}


def sanitize(text: str) -> str:
    """Replace glyphs xhtml2pdf cannot draw. / 替换 PDF 无法绘制的字形。"""
    for src, dst in ASCII_MAP.items():
        text = text.replace(src, dst)
    return text


def build(md_path: str, out_path: str) -> int:
    """Render Markdown to PDF and return the page count. / 渲染并返回页数。"""
    with open(md_path) as fh:
        text = fh.read()
    # Markdown tables need the extension; fenced code is also used in the report.
    # Markdown 表格需要该扩展；报告里也有围栏代码块。
    body = markdown.markdown(sanitize(text), extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><style>{FONT_FACE}{CSS}</style></head><body>{body}</body></html>"
    with open(out_path, "wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF build failed with {result.err} error(s)")
    return len(PdfReader(out_path).pages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="report/report.md")
    ap.add_argument("--out", default="report/report.pdf")
    ap.add_argument("--limit", type=int, default=PAGE_LIMIT)
    args = ap.parse_args()
    print(f"[font] {FONT_PATH or 'built-in (may miss glyphs)'}")
    pages = build(args.md, args.out)
    verdict = "OK" if pages <= args.limit else "OVER LIMIT"
    print(f"[report] {args.md} -> {args.out}  pages={pages}  limit={args.limit}  {verdict}")
    if pages > args.limit:
        sys.exit(f"page limit exceeded: {pages} > {args.limit}")


if __name__ == "__main__":
    main()
