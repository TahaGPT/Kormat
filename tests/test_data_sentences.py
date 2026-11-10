from kormat.data.sentences import ensure_sentences, parse_numbered_note, split_free_text


def test_parse_numbered_with_continuation():
    assert parse_numbered_note("0. First line\ncontinues here.\n1. Second.\n2. Third 3. not an id") == {
        0: "First line continues here.", 1: "Second.", 2: "Third 3. not an id"}


def test_split_free_text_abbrev_and_bullets():
    d = split_free_text("Pt seen by Dr. Smith. Has HTN.\n- Takes aspirin 81 mg daily.\n- No allergies.")
    assert list(d) == list(range(len(d)))
    assert d[0] == "Pt seen by Dr. Smith."
    assert "Takes aspirin 81 mg daily." in d.values() and all(not v.startswith("-") for v in d.values())


def test_ensure_dispatch():
    assert ensure_sentences("0. A.\n1. B.") == {0: "A.", 1: "B."}
    assert list(ensure_sentences("A b. C d.")) == [0, 1]
