"""
Reads the recall@k JSON output and fails (exit code 1) if coverage@5
drops below the threshold. Used as the CI quality gate.
"""
import json
import sys
from pathlib import Path

THRESHOLD = 0.85
K = 5

recall_files = sorted(Path(__file__).parent.glob("recall_at_k*.json"), key=lambda p: p.stat().st_mtime)
if not recall_files:
    print("ERROR: no recall_at_k_*.json file found. Run recall_at_k.py first.")
    sys.exit(1)

recall_file = recall_files[-1]
data = json.loads(recall_file.read_text())

scores = [
    anchor["scores"][str(K)]["coverage"]
    for anchor in data["per_anchor"]
    if str(K) in anchor["scores"]
]

coverage = sum(scores) / len(scores)
print(f"recall@{K} coverage: {coverage:.3f} (threshold: {THRESHOLD})")

if coverage < THRESHOLD:
    print(f"FAIL: {coverage:.3f} < {THRESHOLD}")
    sys.exit(1)

print("PASS")
