"""Builds docs/Antares_Language_Reference.pdf from docs/src/manual.md.

Needs Python 3 with `markdown`, `pygments` and `pymupdf`, and Google Chrome
or Microsoft Edge (used headless to lay out and print the pages).

    python docs/src/build.py
"""

import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # pymupdf
import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import RegexLexer, bygroups, include, words
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Text

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
ROOT = DOCS.parent
OUT = DOCS / "Antares_Language_Reference.pdf"
VERSION = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "Cargo.toml").read_text(), re.M).group(1)
DATE = "November 2026"

KEYWORDS = """and as ask by change descend each else give in learn model not or repeat
show skip stop to try use using when while with xor""".split()
CONSTANTS = ["yes", "no", "nothing"]
BUILTINS = """zeros ones rand randn seed range shape reshape t sum mean max min argmax softmax
onehot grad exp log tanh sigmoid relu sqrt abs sin cos floor round trunc clip sum_rows rows
compact expand screlu sum_product screlu_dot update_rows read_binary input_waiting len int
ints popcount lowest_bit highest_bit div rand_bits push map filter reduce keys values has join
split sort reverse slice upper lower trim replace lines text number kind read write input clock
fail check within json to_json save load exists program_folder key wait clear draw args env append
delete files now fixed pad code char insert remove find unique sort_by""".split()


class AntaresLexer(RegexLexer):
    name = "Antares"
    tokens = {
        "root": [
            (r"#.*$", Comment.Single),
            (r"\s+", Text),
            (r'"', String, "text"),
            (r"(to)(\s+)([A-Za-z_]\w*)", bygroups(Keyword, Text, Name.Function)),
            (r"(model|learn)(\s+)([A-Za-z_]\w*)", bygroups(Keyword, Text, Name.Variable.Magic)),
            (r"0[xX][0-9A-Fa-f_]+", Number.Hex),
            (r"\d[\d_]*(\.\d+)?([eE][+-]?\d+)?", Number),
            (words(KEYWORDS, suffix=r"\b"), Keyword),
            (words(CONSTANTS, suffix=r"\b"), Keyword.Constant),
            (words(BUILTINS, suffix=r"\b(?=\s*\()"), Name.Builtin),
            (r"(adam|sgd|number|list)\b", Name.Builtin.Pseudo),
            (r"[A-Za-z_]\w*", Name),
            (r"->|=>", Operator.Word),
            (r"==|!=|<=|>=|<<|>>|[-+*/%^@<>=&|~]", Operator),
            (r"[()\[\]{},.:]", Punctuation),
        ],
        "text": [
            (r"\\.", String.Escape),
            (r"\{", String.Interpol, "interp"),
            (r'"', String, "#pop"),
            (r'[^"\\{]+', String),
        ],
        "interp": [
            (r"\}", String.Interpol, "#pop"),
            include("root"),
        ],
    }


LEXER = AntaresLexer()
FORMATTER = HtmlFormatter(nowrap=True, classprefix="tk-")


def highlight_code(src: str) -> str:
    return highlight(src, LEXER, FORMATTER).rstrip("\n")


# ---------------------------------------------------------------------------
# Markdown to HTML


def convert(md_text: str):
    chapters = re.split(r"(?m)^(?=# .*\{#[\w-]+[^}]*\}\s*$)", md_text)
    chapters = [c for c in chapters if c.strip()]
    body, toc = [], []
    chap_no, app_no = 0, 0
    for chunk in chapters:
        m = re.match(r"# (.+?)\s*\{#([\w-]+)(\s+\.appendix)?\}\s*\n", chunk)
        title, cid, appendix = m.group(1), m.group(2), bool(m.group(3))
        rest = chunk[m.end():]
        if appendix:
            label = "Appendix " + chr(ord("A") + app_no)
            prefix = chr(ord("A") + app_no)
            app_no += 1
        else:
            chap_no += 1
            label = f"Chapter {chap_no}"
            prefix = str(chap_no)
        md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists"])
        inner = md.convert(rest)
        inner = process_code(inner)
        inner = process_callouts(inner)
        sections = []
        sec_no = 0

        def number_h2(mm):
            nonlocal sec_no
            sec_no += 1
            text = mm.group(1)
            sid = f"{cid}-{sec_no}"
            num = f"{prefix}.{sec_no}"
            sections.append((num, strip_tags(text), sid))
            return f'<h2 id="{sid}"><span class="secno">{num}</span>{text}</h2>'

        inner = re.sub(r"<h2>(.*?)</h2>", number_h2, inner)
        inner = re.sub(r"<h3>(.*?)</h3>", r"<h3>\1</h3>", inner)
        toc.append((label, title, cid, sections, appendix))
        body.append(
            f'<section class="chapter{" appendix" if appendix else ""}" id="{cid}">'
            f'<div class="chap-head"><div class="chap-label">{label}</div>'
            f'<h1>{html.escape(title)}</h1></div>{inner}</section>'
        )
    return "\n".join(body), toc


def strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s))


CODE_RE = re.compile(r'<pre><code(?: class="language-(\w+)")?>(.*?)</code></pre>', re.S)
LABELS = {"output": "Output", "shell": "Terminal", "session": "Interactive session", "grammar": "Grammar", "diagram": None}


def process_code(s: str) -> str:
    def repl(m):
        lang = m.group(1) or "antares"
        src = html.unescape(m.group(2)).rstrip("\n")
        lines = src.count("\n") + 1
        brk = " keep" if lines <= 32 else ""
        if lang == "antares":
            return f'<pre class="code{brk}"><code>{highlight_code(src)}</code></pre>'
        label = LABELS.get(lang)
        body = html.escape(src)
        if lang == "shell":
            body = "\n".join(f'<span class="prompt">$</span> {line}' if line.strip() else line for line in body.split("\n"))
        tag = f'<span class="blk-label">{label}</span>' if label else ""
        return f'<pre class="blk {lang}{brk}">{tag}<code>{body}</code></pre>'

    return CODE_RE.sub(repl, s)


def process_callouts(s: str) -> str:
    def repl(m):
        inner = m.group(1)
        kind = "tip" if "<strong>Tip." in inner else "note"
        return f'<aside class="callout {kind}">{inner}</aside>'

    return re.sub(r"<blockquote>\s*(.*?)\s*</blockquote>", repl, s, flags=re.S)


# ---------------------------------------------------------------------------
# Page furniture


def toc_html(toc, pages):
    rows = []
    for label, title, cid, sections, appendix in toc:
        short = label.replace("Chapter ", "").replace("Appendix ", "")
        rows.append(
            f'<li class="toc-ch"><a href="#{cid}"><span class="toc-num">{short}</span>'
            f'<span class="toc-title">{html.escape(title)}</span>'
            f'<span class="toc-pg">{pages.get(cid, "00")}</span></a></li>'
        )
        for num, stitle, sid in sections:
            rows.append(
                f'<li class="toc-sec"><a href="#{sid}"><span class="toc-num">{num}</span>'
                f'<span class="toc-title">{html.escape(stitle)}</span>'
                f'<span class="toc-pg">{pages.get(sid, "00")}</span></a></li>'
            )
    return '<section class="toc"><h1 class="toc-h">Contents</h1><ol>' + "".join(rows) + "</ol></section>"


COVER_CODE = highlight_code('''learn slope = 0
learn offset = 0

repeat 400 {
    loss = mean((slope * x + offset - y) ^ 2)
    descend loss by 0.05
}

model claude = "anthropic:claude-opus-5-5"
count = ask claude "How many moons does Mars have?" as number''')


def cover_html():
    return f'''<section class="cover">
  <div class="star"></div>
  <div class="cover-top">
    <div class="cover-kicker">Programming Language Reference</div>
    <div class="cover-title">Antares</div>
    <div class="cover-sub">The Language Reference and User Guide</div>
    <div class="cover-tag">A small programming language for AI. Tensors, automatic
    differentiation and language-model calls are part of the language itself.</div>
  </div>
  <pre class="cover-code"><code>{COVER_CODE}</code></pre>
  <div class="cover-foot">
    <div><span class="cf-k">Version</span>{VERSION}</div>
    <div><span class="cf-k">Edition</span>{DATE}</div>
    <div><span class="cf-k">Implementation</span>Rust, zero dependencies</div>
  </div>
</section>'''


def page_html(body, toc, pages):
    css = (HERE / "style.css").read_text(encoding="utf-8")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Antares Language Reference</title>
<style>{css}</style></head>
<body>{cover_html()}{toc_html(toc, pages)}{body}</body></html>'''


# ---------------------------------------------------------------------------
# Printing


def find_browser():
    for p in [
        os.environ.get("CHROME", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ]:
        if p and Path(p).exists():
            return p
    for name in ["google-chrome", "chromium", "chromium-browser", "chrome", "msedge"]:
        found = shutil.which(name)
        if found:
            return found
    sys.exit("Couldn't find Chrome or Edge. Set the CHROME environment variable to its path.")


def print_pdf(browser, html_text, pdf_path, work):
    src = Path(work) / "manual.html"
    src.write_text(html_text, encoding="utf-8")
    profile = Path(work) / "profile"
    subprocess.run(
        [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}", src.as_uri()],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
    )


def heading_pages(pdf_path, ids):
    """Physical page number (1-based) of each heading, read from the TOC's links."""
    doc = fitz.open(pdf_path)
    found = {}
    for page in doc:
        for link in page.get_links():
            name = link.get("nameddest")
            if name in ids and link.get("page", -1) >= 0:
                found[name] = link["page"] + 1
    doc.close()
    missing = [i for i in ids if i not in found]
    if missing:
        sys.exit(f"No page found for {len(missing)} headings, e.g. {missing[:3]}")
    return found


def main():
    body, toc = convert((HERE / "manual.md").read_text(encoding="utf-8"))
    ids = []
    for _, _, cid, sections, _ in toc:
        ids.append(cid)
        ids += [sid for _, _, sid in sections]
    browser = find_browser()
    with tempfile.TemporaryDirectory() as work:
        first = Path(work) / "pass1.pdf"
        print_pdf(browser, page_html(body, toc, {}), first, work)
        pages = heading_pages(first, ids)
        final = Path(work) / "pass2.pdf"
        print_pdf(browser, page_html(body, toc, pages), final, work)
        check = heading_pages(final, ids)
        if check != pages:
            sys.exit("Page numbers moved between passes; run again")

        doc = fitz.open(final)
        outline = []
        for label, title, cid, sections, _ in toc:
            outline.append([1, f"{label}: {title}", pages[cid]])
            outline += [[2, f"{num}  {stitle}", pages[sid]] for num, stitle, sid in sections]
        doc.set_toc([[1, "Cover", 1], [1, "Contents", 2]] + outline)
        doc.set_metadata({
            "title": "Antares Language Reference",
            "subject": f"The Antares programming language, version {VERSION}",
            "keywords": "Antares, programming language, tensors, autodiff, machine learning, LLM",
            "creator": "docs/src/build.py",
            "producer": "Chrome + PyMuPDF",
        })
        doc.save(OUT, garbage=3, deflate=True)
        n = doc.page_count
        doc.close()
    print(f"Wrote {OUT} ({n} pages)")


if __name__ == "__main__":
    main()
