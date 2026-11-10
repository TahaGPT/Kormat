import pytest
import torch

from kormat.decoding import greedy_kv, speculative_greedy
from kormat.modeling import tiny_qwen

V = 1000


def _ids(P, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, V, (1, P), generator=g)


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_identical_draft_full_acceptance(gamma):
    m = tiny_qwen(0, V)
    x = _ids(15)
    ref = greedy_kv(m, x, 33, set(), force_len=True)
    r = speculative_greedy(m, m, x, 33, set(), gamma=gamma, force_len=True)
    assert r.tokens == ref.tokens
    # every full-size round accepts all gamma drafts (the last round is capped by max_new_tokens)
    its = r.stats["per_iter"]
    assert all(i["accepted"] == i["drafted"] for i in its)
    assert r.stats["acceptance_rate"] == 1.0


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
@pytest.mark.parametrize("dseed", [7, 8])
def test_random_draft_still_equals_target(gamma, dseed):
    t, d = tiny_qwen(0, V), tiny_qwen(dseed, V)
    x = _ids(20, seed=dseed)
    ref = greedy_kv(t, x, 40, set(), force_len=True)
    r = speculative_greedy(t, d, x, 40, set(), gamma=gamma, force_len=True)
    assert r.tokens == ref.tokens
    assert r.stats["acceptance_rate"] < 0.5          # unrelated draft rarely agrees
    assert r.stats["iterations"] > 40 // (gamma + 1) - 1


@pytest.mark.parametrize("gamma", [1, 3, 8])
def test_eos_inside_block_and_overshoot(gamma):
    t = tiny_qwen(0, V)
    x = _ids(10)
    full = greedy_kv(t, x, 30, set(), force_len=True).tokens
    eos = full[11]
    ref = greedy_kv(t, x, 30, {eos}).tokens
    for d in (t, tiny_qwen(5, V)):
        r = speculative_greedy(t, d, x, 30, {eos}, gamma=gamma)
        assert r.tokens == ref and r.tokens[-1] == eos
    # overshoot: max_new not a multiple of gamma+1
    for n in (1, 2, 5, 7):
        r = speculative_greedy(t, t, x, n, set(), gamma=gamma, force_len=True)
        assert r.tokens == full[:n]


def test_stats_fields():
    t = tiny_qwen(0, V)
    r = speculative_greedy(t, tiny_qwen(3, V), _ids(9), 20, set(), gamma=4, force_len=True)
    for k in ("gamma", "iterations", "drafted", "accepted", "acceptance_rate", "mean_accepted_per_iter", "per_iter"):
        assert k in r.stats
    assert sum(i["accepted"] + 1 for i in r.stats["per_iter"]) + 1 >= r.n_new
