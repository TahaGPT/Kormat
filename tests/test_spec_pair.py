"""TGA-32: tokenizer/vocab compatibility proof for the draft (0.5B) / target (1.5B) pair. Loads tokenizers + configs
only (no model weights), so it is cheap and needs no model lock."""
import json

from transformers import AutoConfig, AutoTokenizer

from kormat.common.paths import RESULTS
from kormat.modeling import ADAPTER_DIR, DRAFT_ID, TARGET_ID, eos_ids


def test_tokenizer_compat_proof():
    t, d = AutoTokenizer.from_pretrained(TARGET_ID), AutoTokenizer.from_pretrained(DRAFT_ID)
    ft = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))      # tokenizer shipped with the adapter (what the app uses)
    vt, vd = t.get_vocab(), d.get_vocab()
    ct, cd = AutoConfig.from_pretrained(TARGET_ID), AutoConfig.from_pretrained(DRAFT_ID)
    probe = ["Patient note:\n0. A 58-year-old woman, BP 150/90 mmHg.\n", '{"label": "excluded", "evidence": [0, 2]}',
             "<|im_start|>system\nhi<|im_end|>\n"]
    out = {
        "vocab_identical": vt == vd, "n_vocab": len(vt), "config_vocab_size": [ct.vocab_size, cd.vocab_size],
        "added_tokens_identical": t.get_added_vocab() == d.get_added_vocab(),
        "eos": [t.eos_token, d.eos_token], "eos_ids_equal": eos_ids(t) == eos_ids(d), "eos_ids": sorted(eos_ids(t)),
        "chat_template_identical": t.chat_template == d.chat_template,
        "adapter_tokenizer_vocab_identical": ft.get_vocab() == vt,
        "probe_encodings_identical": all(t.encode(s) == d.encode(s) == ft.encode(s) for s in probe),
        "layers": [ct.num_hidden_layers, cd.num_hidden_layers], "hidden": [ct.hidden_size, cd.hidden_size],
        "kv_heads": [ct.num_key_value_heads, cd.num_key_value_heads], "tie_word_embeddings": [ct.tie_word_embeddings, cd.tie_word_embeddings],
    }
    d_dir = RESULTS / "speculative"
    d_dir.mkdir(parents=True, exist_ok=True)
    (d_dir / "pair_tokenizer.json").write_text(json.dumps(out, indent=2))
    assert out["vocab_identical"] and out["config_vocab_size"][0] == out["config_vocab_size"][1]
    assert out["added_tokens_identical"] and out["eos_ids_equal"] and out["probe_encodings_identical"]
    assert out["adapter_tokenizer_vocab_identical"]
