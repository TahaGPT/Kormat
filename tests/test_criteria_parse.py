from kormat.app.criteria_parse import parse_criteria


def test_headers_and_bullets():
    t = "Inclusion Criteria:\n- Age >= 18\n- Type 2 diabetes\n\nExclusion Criteria:\n* Pregnant\n* Heart failure\n  within 6 months\n"
    c = parse_criteria(t)
    assert [x["type"] for x in c] == ["inclusion", "inclusion", "exclusion", "exclusion"]
    assert c[3]["text"] == "Heart failure within 6 months"


def test_numbering():
    c = parse_criteria("INCLUSION CRITERIA\n1. Adults\n2) Consent\nEXCLUSION CRITERIA:\n1. Allergy")
    assert [x["type"] for x in c] == ["inclusion", "inclusion", "exclusion"]
    assert c[1]["text"] == "Consent"


def test_no_headers_default_and_empty():
    assert parse_criteria("Adults over 18\nHas diabetes")[0]["type"] == "inclusion"
    assert len(parse_criteria("Adults over 18\nHas diabetes")) == 2
    assert parse_criteria("  \n") == []


def test_inline_header():
    c = parse_criteria("Exclusion criteria: pregnancy")
    assert c == [{"type": "exclusion", "text": "pregnancy"}]
