"""Compare two trained CPU-model bundles on dev + dev_noisy: how many labels differ, and both accuracies.

  python scripts/compare_models.py SHIPPED.joblib RETRAINED.joblib

Exits non-zero when more than 1% of intents or actions differ (retraining is expected to differ only on a
handful of rows; see EXPERIMENT_LOG P3-1)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from triage.classical import Predictor  # noqa: E402
from triage.data import load_split, project  # noqa: E402

gold = load_split("dev") + load_split("noisy")
rows = [project(r) for r in gold]
a, b = (Predictor(p).predict_many(rows) for p in sys.argv[1:3])
n = len(rows)
diff = {k: sum(x[k] != y[k] for x, y in zip(a, b)) for k in ("intent", "action", "needs_human")}
acc = lambda P, k: round(sum(p[k] == g[k] for p, g in zip(P, gold)) / n, 4)
print(json.dumps({"rows": n, "labels_differing": diff,
                  "max_confidence_difference": round(max(abs(x["confidence"] - y["confidence"]) for x, y in zip(a, b)), 5),
                  "shipped": {"intent_acc": acc(a, "intent"), "action_acc": acc(a, "action")},
                  "retrained": {"intent_acc": acc(b, "intent"), "action_acc": acc(b, "action")}}))
if diff["action"] > 0.01 * n or diff["intent"] > 0.01 * n:
    sys.exit("the retrained model differs from the shipped one on more than 1% of rows")
