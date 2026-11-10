from kormat.app.aggregate import LIKELY_INELIGIBLE, POTENTIAL, UNDETERMINED, aggregate


def R(t, l, i=0):
    return {"criterion_type": t, "label": l, "text": f"c{i}", "index": i}


def test_excluded_wins():
    assert aggregate([R("inclusion", "not enough information"), R("exclusion", "excluded", 1)])["outcome"] == LIKELY_INELIGIBLE


def test_not_included():
    assert aggregate([R("inclusion", "not included"), R("exclusion", "not excluded", 1)])["outcome"] == LIKELY_INELIGIBLE


def test_missing_and_invalid():
    assert aggregate([R("inclusion", "included"), R("exclusion", "not enough information", 1)])["outcome"] == UNDETERMINED
    a = aggregate([R("inclusion", "INVALID")])
    assert a["outcome"] == UNDETERMINED and len(a["missing_information"]) == 1


def test_eligible_and_na_ignored():
    a = aggregate([R("inclusion", "included"), R("exclusion", "not excluded", 1), R("exclusion", "not applicable", 2)])
    assert a["outcome"] == POTENTIAL and len(a["ignored_not_applicable"]) == 1


def test_disqualifier_beats_missing():
    assert aggregate([R("inclusion", "INVALID"), R("inclusion", "not included", 1)])["outcome"] == LIKELY_INELIGIBLE
