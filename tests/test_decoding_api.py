import torch

from kormat.decoding import DecodeResult, greedy_kv, greedy_nocache, hf_generate, speculative_greedy
from kormat.modeling import tiny_qwen


class _Tok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def test_api_fields():
    m, d = tiny_qwen(0, 1000), tiny_qwen(1, 1000)
    x = torch.arange(5, 25)[None]
    for r in (greedy_nocache(m, x, 6, set(), _Tok(), True), greedy_kv(m, x, 6, set(), _Tok(), True),
              speculative_greedy(m, d, x, 6, set(), 3, _Tok(), True), hf_generate(m, x, 6, {999}, _Tok())):
        assert isinstance(r, DecodeResult) and r.n_prompt == 20 and r.n_new == 6 and len(r.tokens) == 6
        assert r.text == " ".join(map(str, r.tokens)) and r.peak_rss_mb > 0
        assert abs(r.tokens_per_s - r.n_new / ((r.prefill_ms + r.decode_ms) / 1000)) < 1e-6 * r.tokens_per_s + 1e-9
