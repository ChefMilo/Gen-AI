import pytest
from pydantic import ValidationError

from evaluation.matching import default_jaccard_min, jaccard, match_findings
from evaluation.universe_record import (
    UniverseRecord,
    measurable_target_ids,
    placebo_record,
    planted_record,
    real_record,
)
from evaluation_helpers import J, finding, ids, placebo_info, planted, truth


def test_default_jaccard_min_reads_config():
    assert default_jaccard_min() == 0.6


def test_exact_match():
    res = match_findings([finding()], planted(truth()), J)
    assert [(m.finding_id, m.spec_id, m.jaccard) for m in res.matches] == [("f1", "t1", 1.0)]
    assert res.outcomes[0].status == "matched"
    assert res.missed_spec_ids == []


def test_jaccard_defaults_to_config():
    assert match_findings([finding()], planted(truth())).jaccard_min == 0.6


def test_jaccard_just_below_at_and_above_threshold():
    t = truth(targets=ids("e", 10))
    # 6 of 10 targets and nothing else: 6/10 = 0.6, exactly at the threshold.
    at = match_findings([finding(event_ids=ids("e", 6))], planted(t), J)
    assert at.matches and at.matches[0].jaccard == pytest.approx(0.6)
    # 6 targets plus 1 non-target: 6/11 just below.
    below = match_findings([finding(event_ids=ids("e", 6) + ["x"])], planted(t), J)
    assert not below.matches
    assert below.outcomes[0].status == "unmatched"
    assert below.outcomes[0].reasons == ["low_jaccard"]
    # 7 of 10: above.
    above = match_findings([finding(event_ids=ids("e", 7))], planted(t), J)
    assert above.matches[0].jaccard == pytest.approx(0.7)


def test_wrong_sign_is_near_miss():
    res = match_findings([finding(effect_car=-0.02)], planted(truth()), J)
    assert not res.matches
    out = res.outcomes[0]
    assert (out.status, out.spec_id, out.reasons) == ("near_miss", "t1", ["wrong_sign"])
    assert res.missed_spec_ids == ["t1"]


def test_short_truth_needs_negative_car():
    t = truth(direction=-1)
    assert match_findings([finding(effect_car=-0.02)], planted(t), J).matches
    assert not match_findings([finding(effect_car=0.02)], planted(t), J).matches


def test_zero_car_has_no_sign():
    res = match_findings([finding(effect_car=0.0)], planted(truth()), J)
    assert res.outcomes[0].reasons == ["wrong_sign"]


def test_not_passed_correction_recorded_as_near_miss():
    res = match_findings([finding(passed=False)], planted(truth()), J)
    assert not res.matches
    out = res.outcomes[0]
    assert (out.status, out.significant, out.reasons) == ("near_miss", False, ["not_significant"])


def test_wrong_sign_and_not_significant_both_listed():
    res = match_findings([finding(effect_car=-0.01, passed=False)], planted(truth()), J)
    assert res.outcomes[0].status == "near_miss"
    assert res.outcomes[0].reasons == ["wrong_sign", "not_significant"]


def test_wrong_event_type_is_not_a_near_miss():
    res = match_findings([finding(event_type="buyback")], planted(truth()), J)
    assert not res.matches
    assert res.outcomes[0].status == "unmatched"
    assert res.outcomes[0].reasons == ["wrong_event_type"]


def test_two_findings_compete_for_one_truth():
    better = finding("f2", event_ids=ids("e", 9))
    worse = finding("f1", event_ids=ids("e", 7))
    res = match_findings([worse, better], planted(truth()), J)
    assert [(m.finding_id, m.spec_id) for m in res.matches] == [("f2", "t1")]
    dup = res.outcomes[0]
    assert (dup.finding_id, dup.status, dup.reasons) == ("f1", "duplicate",
                                                         ["truth_already_matched"])


def test_tie_goes_to_earlier_finding():
    a, b = finding("a", event_ids=ids("e", 8)), finding("b", event_ids=ids("e", 8, start=2))
    res = match_findings([b, a], planted(truth()), J)
    assert [m.finding_id for m in res.matches] == ["b"]
    res = match_findings([a, b], planted(truth()), J)
    assert [m.finding_id for m in res.matches] == ["a"]


def test_one_finding_overlapping_two_truths_matches_one():
    t1 = truth("t1", targets=ids("e", 10))
    t2 = truth("t2", targets=ids("e", 12))  # finding is 10/12 of t2, all of t1
    res = match_findings([finding()], planted(t1, t2), J)
    assert [(m.finding_id, m.spec_id) for m in res.matches] == [("f1", "t1")]
    assert res.missed_spec_ids == ["t2"]


def test_greedy_assignment_lets_second_finding_take_other_truth():
    t1 = truth("t1", targets=ids("e", 10))
    t2 = truth("t2", targets=ids("e", 12))
    f1 = finding("f1", event_ids=ids("e", 10))  # 1.0 on t1, 0.83 on t2
    f2 = finding("f2", event_ids=ids("e", 11))  # 0.91 on t1, 0.92 on t2
    res = match_findings([f1, f2], planted(t1, t2), J)
    assert {(m.finding_id, m.spec_id) for m in res.matches} == {("f1", "t1"), ("f2", "t2")}


def test_unmeasurable_targets_excluded_from_jaccard():
    t = truth(targets=ids("e", 10), unmeasurable=ids("e", 4, start=6))
    assert measurable_target_ids(t) == set(ids("e", 6))
    res = match_findings([finding(event_ids=ids("e", 6))], planted(t), J)
    assert res.matches[0].jaccard == 1.0
    # Reporting the unmeasurable targets too costs overlap: 6/10.
    res = match_findings([finding(event_ids=ids("e", 10))], planted(t), J)
    assert res.matches[0].jaccard == pytest.approx(0.6)


def test_empty_findings():
    res = match_findings([], planted(truth()), J)
    assert res.matches == [] and res.outcomes == []
    assert res.missed_spec_ids == ["t1"] and res.n_truths == 1


def test_finding_with_no_event_ids_does_not_match():
    res = match_findings([finding(event_ids=[])], planted(truth()), J)
    assert res.outcomes[0].jaccard == 0.0 and not res.matches


def test_jaccard_empty_sets():
    assert jaccard(set(), set()) == 0.0


def test_placebo_findings_are_false_discoveries():
    rec = placebo_record("p1", placebo_info(seed=3))
    assert rec.seed == 3
    res = match_findings([finding("a"), finding("b", passed=False), finding("c")], rec, J)
    assert [o.status for o in res.outcomes] == [
        "false_discovery", "not_significant", "false_discovery"
    ]
    assert res.matches == [] and res.n_truths == 0


def test_real_universe_findings_unscored():
    res = match_findings([finding()], real_record("real"), J)
    assert res.outcomes[0].status == "unscored"


def test_duplicate_finding_ids_rejected():
    with pytest.raises(ValueError, match="duplicate finding_id"):
        match_findings([finding("a"), finding("a")], planted(truth()), J)


def test_record_validation():
    with pytest.raises(ValidationError, match="cannot have planted truths"):
        UniverseRecord(universe_id="p", universe_type="placebo", seed=0, truths=[truth()])
    with pytest.raises(ValidationError, match="only for placebo"):
        UniverseRecord(universe_id="u", universe_type="planted", seed=0,
                       placebo_info=placebo_info())
    with pytest.raises(ValidationError, match="duplicate spec_id"):
        planted(truth("t1"), truth("t1"))


def test_planted_record_seed_from_specs():
    t1, t2 = truth("t1"), truth("t2")
    t2 = t2.model_copy(update={"spec": t2.spec.model_copy(update={"seed": 7})})
    assert planted_record("u", [t1]).seed == 0
    with pytest.raises(ValueError, match="pass seed explicitly"):
        planted_record("u", [t1, t2])
    assert planted_record("u", [t1, t2], seed=5).seed == 5
