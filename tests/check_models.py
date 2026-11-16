"""Checks Antares's model support offline, against a pretend model service.

Starts tests/mock_models.py, then runs tests/models.ant with the service's
address in ANTARES_MOCK_URL. Pass --pack or --native to check a built program
instead of the interpreter.

Run from the project folder:  python tests/check_models.py [--pack | --native]
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANTARES = os.path.join(ROOT, "target", "release", "antares.exe" if os.name == "nt" else "antares")


def main():
    # a one-pixel PNG for the image test
    png = os.path.join(ROOT, "tests", "pixel.png")
    if not os.path.exists(png):
        import base64
        open(png, "wb").write(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="))
    server = subprocess.Popen([sys.executable, os.path.join(ROOT, "tests", "mock_models.py")], stdout=subprocess.PIPE, text=True)
    try:
        port = server.stdout.readline().strip()
        env = dict(os.environ, ANTARES_MOCK_URL=f"http://127.0.0.1:{port}")
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("OPENAI_API_KEY", None)
        cmd = [ANTARES, "tests/models.ant"]
        mode = next((a for a in sys.argv[1:] if a in ("--pack", "--native")), None)
        if mode:
            out = os.path.join(ROOT, "tests", "models_built" + (".exe" if os.name == "nt" else ""))
            subprocess.run([ANTARES, "build", "tests/models.ant", out, mode], cwd=ROOT, check=True)
            cmd = [out]
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
        if mode:
            os.remove(out)
        sys.stdout.write(p.stdout)
        sys.stderr.write(p.stderr)
        return p.returncode
    finally:
        server.terminate()


if __name__ == "__main__":
    sys.exit(main())
