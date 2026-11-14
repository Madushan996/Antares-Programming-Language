"""Speed comparison: Antares vs Python vs Node on the same small programs.

Each program is run several times and the best time is kept, because single
runs on a busy machine vary a lot. Times are measured inside each program,
so interpreter start-up is not counted.

Run from the project folder:  python bench/bench.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

RUNS = 7
HERE = os.path.dirname(os.path.abspath(__file__))
ANTARES = os.path.join(HERE, "..", "target", "release", "antares.exe" if os.name == "nt" else "antares")

TESTS = {
    "fib(30): 1.3M function calls": {
        "ant": """
to fib(n) { when n < 2 { n } else { fib(n - 1) + fib(n - 2) } }
t = clock()
fib(30)
show clock() - t
""",
        "py": """
import time
def fib(n): return n if n < 2 else fib(n - 1) + fib(n - 2)
t = time.perf_counter(); fib(30); print(time.perf_counter() - t)
""",
        "js": """
function fib(n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }
let t = performance.now(); fib(30); console.log((performance.now() - t) / 1000);
""",
    },
    "loop: 5M additions": {
        "ant": """
t = clock()
x = 0
repeat 5000000 { x = x + 1 }
show clock() - t
""",
        "py": """
import time
t = time.perf_counter()
x = 0
for i in range(5000000): x = x + 1
print(time.perf_counter() - t)
""",
        "js": """
let t = performance.now(); let x = 0;
for (let i = 0; i < 5000000; i++) x = x + 1;
console.log((performance.now() - t) / 1000);
""",
    },
    "while loop with a condition: 3M rounds": {
        "ant": """
t = clock()
i = 0
total = 0
while i < 3000000 {
    when i % 3 == 0 { total = total + i }
    i = i + 1
}
show clock() - t
""",
        "py": """
import time
t = time.perf_counter()
i = 0; total = 0
while i < 3000000:
    if i % 3 == 0: total = total + i
    i = i + 1
print(time.perf_counter() - t)
""",
        "js": """
let t = performance.now(); let i = 0, total = 0;
while (i < 3000000) { if (i % 3 === 0) total = total + i; i = i + 1; }
console.log((performance.now() - t) / 1000);
""",
    },
    "list reads: 3M lookups": {
        "ant": """
xs = ["a", "b", "c", "d", "e", "f", "g", "h"]
t = clock()
i = 0
n = 0
while i < 3000000 {
    when xs[i % 8] == "c" { n = n + 1 }
    i = i + 1
}
show clock() - t
""",
        "py": """
import time
xs = ["a", "b", "c", "d", "e", "f", "g", "h"]
t = time.perf_counter()
i = 0; n = 0
while i < 3000000:
    if xs[i % 8] == "c": n = n + 1
    i = i + 1
print(time.perf_counter() - t)
""",
        "js": """
const xs = ["a", "b", "c", "d", "e", "f", "g", "h"];
let t = performance.now(); let i = 0, n = 0;
while (i < 3000000) { if (xs[i % 8] === "c") n = n + 1; i = i + 1; }
console.log((performance.now() - t) / 1000);
""",
    },
    "matrix multiply 300x300, three times": {
        "ant": """
seed(1)
a = randn(300, 300)
t = clock()
b = a @ a @ a @ a
show clock() - t
""",
        "py": """
import time, random
random.seed(1)
n = 300
a = [[random.gauss(0, 1) for _ in range(n)] for _ in range(n)]
def mm(x, y):
    yt = list(zip(*y))
    return [[sum(p * q for p, q in zip(row, col)) for col in yt] for row in x]
t = time.perf_counter(); b = mm(mm(mm(a, a), a), a); print(time.perf_counter() - t)
""",
        "js": """
const n = 300;
const a = Array.from({length: n}, () => Float64Array.from({length: n}, () => Math.random()));
function mm(x, y) {
  const o = Array.from({length: n}, () => new Float64Array(n));
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) { const v = x[i][k]; for (let j = 0; j < n; j++) o[i][j] += v * y[k][j]; }
  return o;
}
let t = performance.now(); const b = mm(mm(mm(a, a), a), a); console.log((performance.now() - t) / 1000);
""",
    },
}


def best(cmd, src, ext):
    with tempfile.NamedTemporaryFile("w", suffix=ext, delete=False, encoding="utf-8") as f:
        f.write(src)
        path = f.name
    times = []
    try:
        for _ in range(RUNS):
            out = subprocess.run(cmd + [path], capture_output=True, text=True)
            if out.returncode != 0:
                return None
            times.append(float(out.stdout.strip().splitlines()[-1]))
    finally:
        os.unlink(path)
    return min(times)


def main():
    runners = [("Antares", [ANTARES], "ant", ".ant"), ("Python", [sys.executable], "py", ".py")]
    if shutil.which("node"):
        runners.append(("Node", ["node"], "js", ".js"))
    print(f"best of {RUNS} runs, seconds (lower is better)\n")
    print(f"{'test':42}" + "".join(f"{name:>10}" for name, *_ in runners) + "   Antares vs Python")
    for title, progs in TESTS.items():
        row = [best(cmd, progs[key], ext) for _, cmd, key, ext in runners]
        cells = "".join(f"{(f'{t:.3f}' if t is not None else 'failed'):>10}" for t in row)
        ratio = f"{row[1] / row[0]:.1f}x faster" if row[0] and row[1] and row[1] >= row[0] else (
            f"{row[0] / row[1]:.1f}x slower" if row[0] and row[1] else "")
        print(f"{title:42}{cells}   {ratio}")


if __name__ == "__main__":
    main()
