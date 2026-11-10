from kormat.eval.evidence import evidence_report, prf_pair

# Toy records:
# r1 G={1,2} P={2,3}: tp1 fp1 fn1 -> P=R=.5 F1=.5
# r2 G={} P={}      : 1,1,1 (contributes nothing to micro)
# r3 G={} P={4}     : 0,1,0 ; fp1
# r4 G={5} P={}     : 1,0,0 ; fn1
# micro: tp1 fp2 fn2 -> P=1/3 R=1/3 F1=1/3
# macro_all: P=(.5+1+0+1)/4=.625 R=(.5+1+1+0)/4=.625 F1=(.5+1+0+0)/4=.375
# macro_nonempty_gold (r1,r4): P=.75 R=.25 F1=.25
G = [[1, 2], [], [], [5]]
P = [[2, 3], [], [4], []]
T = ["inclusion", "inclusion", "exclusion", "exclusion"]
L = ["included", "not applicable", "not enough information", "excluded"]


def test_pair_rules():
    assert prf_pair(set(), set()) == (1, 1, 1)
    assert prf_pair(set(), {1}) == (0, 1, 0)
    assert prf_pair({1}, set()) == (1, 0, 0)
    assert prf_pair({1, 2}, {1, 2}) == (1, 1, 1)
    assert prf_pair({1}, {2}) == (0, 0, 0)


def test_report():
    r = evidence_report(G, P, T, L)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 2, 2)
    assert abs(r["micro"]["f1"] - 1 / 3) < 1e-9
    assert abs(r["macro_all"]["precision"] - .625) < 1e-9
    assert abs(r["macro_all"]["recall"] - .625) < 1e-9
    assert abs(r["macro_all"]["f1"] - .375) < 1e-9
    assert abs(r["macro_nonempty_gold"]["precision"] - .75) < 1e-9
    assert abs(r["macro_nonempty_gold"]["f1"] - .25) < 1e-9
    assert r["cells"] == {"gold_empty_pred_empty": 1, "gold_empty_pred_nonempty": 1,
                          "gold_nonempty_pred_empty": 1, "gold_nonempty_pred_nonempty": 1}
    assert r["by_type"]["exclusion"]["n"] == 2 and r["by_gold_label"]["included"]["micro"]["f1"] == 0.5


def test_all_empty_micro_zero_not_crash():
    r = evidence_report([[]], [[]], ["inclusion"], ["not applicable"])
    assert r["micro"]["f1"] == 0.0 and r["macro_all"]["f1"] == 1.0
