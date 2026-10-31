import json
from datetime import datetime

from rag_helpers import make_event

from rag.index import HybridIndex
from rag.text_features import FeatureSpec, classify, label_events

SPEC = FeatureSpec(
    event_type="dividend",
    query="reason for the dividend",
    labels={"increase_confidence": "raise tied to confidence", "one_off": "special payment"},
)
PASSAGES = "AAA Corp declared a dividend,  reflecting\nconfidence in future cash flow."


def reply(label, quote):
    return json.dumps({"label": label, "quote": quote})


class ScriptedAsk:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.replies[len(self.calls) - 1]


def test_valid_reply_becomes_a_text_feature():
    ask = ScriptedAsk("```json\n" + reply("increase_confidence", "reflecting confidence") + "\n```")
    feature = classify("dividend_framing", SPEC, PASSAGES, ask)
    assert feature.label == "increase_confidence"
    assert feature.quote == "reflecting confidence"  # whitespace differences are tolerated
    system, user = ask.calls[0]
    assert "data" in system and "instructions" in system
    assert f"<passages>\n{PASSAGES}\n</passages>" in user
    assert "increase_confidence: raise tied to confidence" in user


def test_one_retry_with_the_error_message():
    ask = ScriptedAsk(
        reply("growth", "reflecting confidence"), reply("one_off", "declared a dividend")
    )
    feature = classify("dividend_framing", SPEC, PASSAGES, ask)
    assert feature.label == "one_off"
    assert "growth" in ask.calls[1][1] and "rejected" in ask.calls[1][1]


def test_invented_quote_fails_after_the_retry():
    ask = ScriptedAsk(reply("one_off", "a special dividend"), "not json at all")
    assert classify("dividend_framing", SPEC, PASSAGES, ask) is None
    assert len(ask.calls) == 2


def test_label_events_labels_each_filing_once_and_keeps_other_events(docs, embedder):
    index = HybridIndex.build(docs, embedder, max_chars=500, candidates=50, rrf_k=60)
    events = [
        make_event("0001-16-000001-dividend-1", datetime(2016, 3, 1, 9), {}),
        make_event("0001-16-000001-dividend-2", datetime(2016, 3, 1, 9), {}),
        make_event("0001-18-000002-buyback-1", datetime(2018, 5, 2, 9), {}, event_type="buyback"),
    ]
    ask = ScriptedAsk(reply("increase_confidence", "reflecting confidence in future cash flow"))

    labelled = label_events(events, index, {"dividend_framing": SPEC}, ask, n_passages=3)

    assert len(ask.calls) == 1
    assert "AAA Corp declared a quarterly cash dividend" in ask.calls[0][1]
    assert [e.event_id for e in labelled] == [e.event_id for e in events]
    for event in labelled[:2]:
        assert event.text_features["dividend_framing"].label == "increase_confidence"
    assert labelled[2].text_features == {}
