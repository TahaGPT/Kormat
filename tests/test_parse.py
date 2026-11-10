from kormat.prompting.parse import parse_output

V = {1, 2, 3, 4}


def p(raw, t="inclusion", v=V):
    return parse_output(raw, t, v)


def test_clean():
    r = p('{"label": "included", "evidence": [3, 1]}')
    assert r == {"label": "included", "evidence": [1, 3], "status": "ok", "hallucinated_ids": []}


def test_fence_trailing_think():
    raw = '<think>hmm {"label":"x"}</think>\n```json\n{"label": "not included", "evidence": [2]}\n```\nHope this helps'
    r = p(raw)
    assert r["label"] == "not included" and r["evidence"] == [2] and r["status"] == "ok"


def test_single_quotes_lenient():
    r = p("{'label': 'excluded', 'evidence': [1]}", "exclusion")
    assert r["label"] == "excluded" and r["status"] == "lenient"


def test_synonyms():
    assert p('{"label": "Insufficient information", "evidence": []}')["label"] == "not enough information"
    r = p('{"label": "N/A", "evidence": []}')
    assert r["label"] == "not applicable" and r["status"] == "lenient"


def test_type_mismatch_invalid():
    assert p('{"label": "excluded", "evidence": [1]}', "inclusion")["label"] == "INVALID"
    assert p('{"label": "included", "evidence": [1]}', "exclusion")["label"] == "INVALID"


def test_garbage_never_real_class():
    for raw in ["", "I think it is included", "not json", "{}", '{"label": 5}', '{"label": "maybe"}', None]:
        r = p(raw)
        assert r["label"] == "INVALID" and r["status"] == "invalid" and r["evidence"] == []


def test_hallucinated_ids_and_truncation():
    r = p('{"label": "included", "evidence": [2, 9, "3", 2]}')
    assert r["evidence"] == [2, 3] and r["hallucinated_ids"] == [9]
    r = p('{"label": "included", "evidence": [1, 2')  # truncated -> regex fallback
    assert r["label"] == "invalid" or r["status"] in ("lenient", "invalid")
