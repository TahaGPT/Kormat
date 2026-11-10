import numpy as np
from sklearn.metrics import f1_score

from kormat.eval.classification import bootstrap_ci, classification_report, plot_confusion
from kormat.prompting.labels import ALL_LABELS

# Toy (inclusion only): gold = [inc, inc, notinc, nei], pred = [inc, notinc, notinc, INVALID]
# included: tp1 fp0 fn1 -> P1 R.5 F1 2/3 ; not included: tp1 fp1 fn0 -> P.5 R1 F1 2/3
# nei: tp0 fn1 -> F1 0 ; not applicable: support 0 -> F1 0 (zero_division=0)
# macro = (2/3+2/3+0+0)/4 = 1/3 ; accuracy = 2/4
G = ["included", "included", "not included", "not enough information"]
P = ["included", "not included", "not included", "INVALID"]
T = ["inclusion"] * 4


def test_hand_computed():
    r = classification_report(G, P, T)["inclusion"]
    assert abs(r["macro_f1"] - 1 / 3) < 1e-9
    assert r["accuracy"] == 0.5
    assert abs(r["per_class"]["included"]["f1"] - 2 / 3) < 1e-9
    assert r["per_class"]["not enough information"]["support"] == 1
    assert r["confusion"]["labels"][-1] == "INVALID"
    assert r["confusion"]["matrix"][0] == [1, 1, 0, 0, 0]
    assert r["confusion"]["matrix"][2] == [0, 0, 0, 0, 1]


def test_sklearn_crosscheck_and_mixed():
    rng = np.random.default_rng(0)
    g = list(rng.choice(ALL_LABELS, 60))
    p_ = list(rng.choice(list(ALL_LABELS) + ["INVALID"], 60))
    t = ["inclusion" if x in ("included", "not included") else "exclusion" for x in g]
    rep = classification_report(g, p_, t)
    assert abs(rep["overall"]["macro_f1"] - f1_score(g, p_, average="macro", labels=list(ALL_LABELS), zero_division=0)) < 1e-12
    assert abs(rep["overall"]["weighted_f1"] - f1_score(g, p_, average="weighted", labels=list(ALL_LABELS), zero_division=0)) < 1e-12
    assert rep["overall"]["n"] == 60


def test_bootstrap_and_plot(tmp_path):
    ci = bootstrap_ci(G * 10, G * 10, "accuracy", n=50)
    assert ci["point"] == 1.0 and ci["lo"] == 1.0 and ci["hi"] == 1.0
    ci2 = bootstrap_ci(G * 10, P * 10, "macro_f1", n=100)
    assert ci2["lo"] <= ci2["point"] <= ci2["hi"] or ci2["lo"] <= ci2["hi"]
    assert bootstrap_ci(G, P, "accuracy", n=50) == bootstrap_ci(G, P, "accuracy", n=50)
    plot_confusion(classification_report(G, P, T)["inclusion"], tmp_path / "c.pdf")
    assert (tmp_path / "c.pdf").stat().st_size > 0
