"""Render module_summary.md to module_summary.pdf (needs: pip install markdown weasyprint)."""
import re
from pathlib import Path

import markdown
import weasyprint

ROOT = Path(__file__).resolve().parent.parent
CSS = """@page { size: Letter; margin: 2.0cm 2.1cm; @bottom-center { content: counter(page); font: 9pt serif; color:#555 } }
body { font-family: "Liberation Serif", "DejaVu Serif", serif; font-size: 10.8pt; line-height: 1.35; color:#111; }
h1 { font-size: 17pt; margin: 0 0 4pt; text-align:center; } h1 + p { text-align:center; margin-top:0; font-size:10pt }
h2 { font-size: 13pt; margin: 14pt 0 5pt; border-bottom: 0.6pt solid #999; padding-bottom: 2pt; page-break-after: avoid; }
p { margin: 5pt 0; text-align: justify; } li { margin: 2pt 0; }
code { font-family: "Liberation Mono", "DejaVu Sans Mono", monospace; font-size: 8.8pt; background:#f3f3f3; }
table { border-collapse: collapse; width: 100%; margin: 7pt 0; font-size: 9pt; } tr { page-break-inside: avoid; }
th, td { border: 0.6pt solid #888; padding: 3pt 5pt; vertical-align: top; text-align:left; } th { background: #e9edf2; }
figure { margin: 8pt 0; text-align:center; page-break-inside: avoid; } figure img { width: 100%; }
figcaption { font-size: 9pt; font-style: italic; color:#333; }
.refs p { padding-left: 2.2em; text-indent: -2.2em; text-align:left; } a { color:#1a4f8b; text-decoration:none; }"""
src = (ROOT / "module_summary.md").read_text(encoding="utf-8")
html = markdown.markdown(src, extensions=["tables", "sane_lists"])
html = re.sub(r'<p><img alt="([^"]+)" src="([^"]+)" ?/?></p>',
              r'<figure><img src="\2" alt="\1"><figcaption>Figure 1. \1</figcaption></figure>', html)
html = re.sub(r'(?<![">=])(https?://[^\s<]+?)(?=[\s<]|$)', r'<a href="\1">\1</a>', html)
html = html.replace("<h2>References</h2>", '<h2>References</h2><div class="refs">') + "</div>"
doc = f'<html><head><meta charset="utf-8"><title>NeuroGuard AI - Integrative Industry Synthesis</title></head><body>{html}</body></html>'
weasyprint.HTML(string=doc, base_url=str(ROOT)).write_pdf(ROOT / "module_summary.pdf", stylesheets=[weasyprint.CSS(string=CSS)])
print("wrote module_summary.pdf")
