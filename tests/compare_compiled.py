"""Checks that compiled programs behave exactly like interpreted ones.

For each program: run it with the interpreter, compile it with `antares build`,
run the result, and compare what both print and how both exit.

Run from the project folder:  python tests/compare_compiled.py [files...]
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANTARES = os.path.join(ROOT, "target", "release", "antares.exe" if os.name == "nt" else "antares")
DEFAULT = [
    "tests/check.ant", "tests/features.ant", "tests/engine.ant", "tests/integers.ant", "tests/counters.ant",
    "examples/hello.ant", "examples/line.ant", "examples/xor.ant", "examples/ask.ant",
]


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout, p.stderr


def main():
    files = sys.argv[1:] or DEFAULT
    ok = True
    for f in files:
        exe = os.path.join(ROOT, os.path.splitext(f)[0] + (".exe" if os.name == "nt" else ""))
        want = run([ANTARES, f])
        code, _, err = run([ANTARES, "build", f])
        if code != 0:
            ok = False
            print(f"FAIL {f}: did not compile\n{err[:3000]}")
            continue
        got = run([exe])
        os.remove(exe)
        if got[0] == want[0] and got[1] == want[1]:
            print(f"ok   {f}: same output ({len(got[1].splitlines())} lines, exit {got[0]})")
        else:
            ok = False
            print(f"FAIL {f}: interpreter exit {want[0]}, compiled exit {got[0]}")
            a, b = want[1].splitlines(), got[1].splitlines()
            for i in range(max(len(a), len(b))):
                x = a[i] if i < len(a) else "<nothing>"
                y = b[i] if i < len(b) else "<nothing>"
                if x != y:
                    print(f"  line {i + 1}: interpreter: {x}\n  line {i + 1}: compiled:    {y}")
                    break
            print("  compiled stderr:", got[2].strip()[:600])
    print("compiled and interpreted programs agree" if ok else "DIFFERENCES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
