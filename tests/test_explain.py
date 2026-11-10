from kormat.eval.explain import build_explanation

S = {3: "Patient has diabetes.", 4: "No cancer."}


def test_cited_only():
    e = build_explanation("not excluded", [3], S, "exclusion", "c")
    assert e == "Sentence 3 states: 'Patient has diabetes.'. On that basis the exclusion criterion is judged 'not excluded'."
    assert "No cancer" not in e


def test_empty_and_special():
    assert build_explanation("not enough information", [], S, "inclusion", "c").startswith("No sentence in the note addresses")
    assert "not applicable" in build_explanation("not applicable", [], S, "inclusion", "c")
    assert "could not be parsed" in build_explanation("INVALID", [], S, "inclusion", "c")
    assert "99" not in build_explanation("included", [99], S, "inclusion", "c")


def test_error_tags():
    from kormat.eval.errors import tag_row
    r = {"gold_label": "included", "parsed_label": "not enough information", "gold_ids": [1], "parsed_ids": [], "hallucinated_ids": []}
    t = tag_row(r)
    assert "over_abstain" in t and "missed_evidence" in t
    assert tag_row({"gold_label": "included", "parsed_label": "included", "gold_ids": [1], "parsed_ids": [1], "hallucinated_ids": []}) == []
