"""Runs every ```antares example in a document and compares it with the
```output box after it, so the beginner's book stays right as Antares changes.

  python docs/src/check_examples.py                 the book (docs/src/tutorial.md)
  python docs/src/check_examples.py file.md ...     other documents

Uses target/release/antares if it has been built, otherwise `antares` on the PATH.

Markers on the line before a code box:
  <!-- norun -->   not run (needs typing, keys, a network or other files)
  <!-- error -->   expected to stop with a problem; the output box holds the message
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BUILT = ROOT / "target" / "release" / ("antares.exe" if os.name == "nt" else "antares")
ANTARES = str(BUILT) if BUILT.exists() else "antares"

FENCE = re.compile(r"^```(\w*)\s*$")


def blocks(text):
    lines = text.split("\n")
    out = []
    i = 0
    marker = None
    while i < len(lines):
        line = lines[i]
        m = FENCE.match(line)
        if m:
            lang = m.group(1) or "antares"
            start = i + 1
            j = start
            while not lines[j].startswith("```"):
                j += 1
            out.append({"lang": lang, "code": "\n".join(lines[start:j]), "line": i + 1, "marker": marker})
            marker = None
            i = j + 1
            continue
        s = line.strip()
        if s.startswith("<!--"):
            marker = s[4:-3].strip()
        elif s:
            marker = None if not s.startswith("<!--") else marker
        i += 1
    return out


def clean(s):
    lines = [l.rstrip() for l in s.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def main():
    bad = 0
    total = 0
    for path in sys.argv[1:] or [str(ROOT / "docs" / "src" / "tutorial.md")]:
        text = open(path, encoding="utf-8").read()
        bs = blocks(text)
        for k, b in enumerate(bs):
            if b["lang"] != "antares" or b["marker"] == "norun":
                continue
            total += 1
            expected = None
            if k + 1 < len(bs) and bs[k + 1]["lang"] == "output":
                # only when nothing but blank lines sits between the boxes
                # lines after this box's closing fence, up to the output box's opening fence
                between = text.split("\n")[b["line"] + b["code"].count("\n") + 2: bs[k + 1]["line"] - 1]
                if all(not l.strip() for l in between):
                    expected = bs[k + 1]["code"]
            with tempfile.TemporaryDirectory() as d:
                src = os.path.join(d, "example.ant")
                open(src, "w", encoding="utf-8").write(b["code"] + "\n")
                p = subprocess.run([ANTARES, src], capture_output=True, text=True, encoding="utf-8",
                                   stdin=subprocess.DEVNULL, cwd=d, timeout=120)
            got = p.stdout + (p.stderr if b["marker"] == "error" else "")
            problem = None
            if b["marker"] == "error":
                if p.returncode == 0:
                    problem = "expected a problem, but it ran"
            elif p.returncode != 0:
                problem = "stopped with a problem:\n" + clean(p.stderr)
            if problem is None and expected is not None and clean(got) != clean(expected):
                problem = "output differs:\n--- expected\n" + clean(expected) + "\n--- got\n" + clean(got)
            if problem:
                bad += 1
                print(f"{path}:{b['line']}: {problem}\n")
    print(f"{total - bad} of {total} examples ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
