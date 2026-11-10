"""Plot the shared training run's logged curves (provenance figure) -> report/figures/training_curves_shared.pdf."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kormat.common.paths import FIGURES, RESULTS

D = RESULTS / "finetune" / "shared_run"
steps = json.loads((D / "log_history.json").read_text())["steps"]
ep = json.loads((D / "val_metrics_by_epoch.json").read_text())["epochs"]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6))
a.plot([s["epoch"] for s in steps], [s["train_loss"] for s in steps], color="#2a6fdb", lw=1.4, label="train loss")
a.plot([e["epoch"] for e in ep], [e["val_loss"] for e in ep], "o--", color="#d9480f", lw=1.2, label="val loss")
a.set_xlabel("epoch"); a.set_ylabel("loss (CE)"); a.legend(frameon=False, fontsize=8)
a.grid(alpha=0.25)
x = [e["epoch"] for e in ep]
b.plot(x, [e["accuracy"] for e in ep], "o-", color="#2a6fdb", label="accuracy")
b.plot(x, [e["macro_f1"] for e in ep], "s-", color="#7b4dff", label="macro-F1")
b.plot(x, [e["evidence_micro_f1"] for e in ep], "^-", color="#2b8a3e", label="evidence micro-F1")
b.set_xticks(x); b.set_xlabel("epoch"); b.set_ylabel("val score"); b.set_ylim(0, 1)
b.legend(frameon=False, fontsize=8); b.grid(alpha=0.25)
fig.tight_layout()
out = FIGURES / "training_curves_shared.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=150)
print(out)
