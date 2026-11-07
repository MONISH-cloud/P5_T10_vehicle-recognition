"""Final submission check. Run before the deadline."""
import re, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
problems = []
for path in ["README.md", "docs/DESIGN.md", "docs/architecture.png", "docs/ABLATION_AND_FAILURES.md", "Dockerfile", "app.py"]:
    if not (root / path).exists():
        problems.append(f"Missing {path}")
evals = list(root.glob("eval_P5_*.md"))
if not evals:
    problems.append("Missing eval_P5_<TeamID>.md")
for path in evals:
    first = path.read_text().splitlines()[0]
    if not re.fullmatch(r"[A-Za-z0-9]+(,\s*[A-Za-z0-9]+)*", first):
        problems.append(f"First line of {path.name} must be comma-separated USNs")
    if "PARTIAL RUN" in path.read_text():
        problems.append(f"{path.name} comes from a limited smoke test")
for path in ["docs/ABLATION_AND_FAILURES.md", "docs/DESIGN.md", "README.md"]:
    if (root / path).exists() and "TEAM TO" in (root / path).read_text():
        problems.append(f"{path} still has a TEAM TO marker")
patterns = re.compile(r"(hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|\"key\"\s*:\s*\"[a-f0-9]{32}\")")
try:
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
except (subprocess.CalledProcessError, FileNotFoundError):
    files = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".json", ".yaml", ".txt"}]
for file in files:
    try:
        if patterns.search((root / file).read_text()):
            problems.append(f"Possible secret in {file}")
    except (UnicodeDecodeError, FileNotFoundError):
        pass
print("\n".join(problems) if problems else "All checks passed.")
sys.exit(1 if problems else 0)
