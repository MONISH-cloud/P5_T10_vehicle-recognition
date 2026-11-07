"""Generate eval_P5_<TeamID>.md and the ablation/failure document from results/metrics.json only."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vehicle_id.reporting import ablation_markdown, eval_markdown

parser = argparse.ArgumentParser()
parser.add_argument("--team-id", required=True)
parser.add_argument("--usns", required=True, help="Comma-separated USNs of all members")
parser.add_argument("--metrics", default="results/metrics.json")
parser.add_argument("--out", default=".")
args = parser.parse_args()
results = json.loads(Path(args.metrics).read_text())
out = Path(args.out)
(out / f"eval_P5_{args.team_id}.md").write_text(eval_markdown(results, args.usns))
(out / "docs").mkdir(exist_ok=True)
target = out / "docs" / "ABLATION_AND_FAILURES.md"
target.write_text(ablation_markdown(results))
print("Wrote", out / f"eval_P5_{args.team_id}.md", "and", target)
print("Now replace every TEAM TO EXPLAIN marker in", target, "after looking at results/failures/.")
