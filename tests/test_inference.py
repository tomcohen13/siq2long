import json

import pandas as pd
import pytest

from config import Columns
from inference import Result, group_by_video, parse, run, summarize, to_frame
from fakes import FakeVLM, make_row


# --- parse -------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        ("0", 0),
        ("3", 3),
        ("Answer: 2", 2),
        ("The answer is 1.", 1),
        (" 2\n", 2),
        ("", None),
        ("none of these", None),
        ("9", None),          # out of range
        ("4", None),
        ("42", 2),            # first *in-range* digit, not the first digit
        ("-1", 1),            # sign is not consumed
    ],
)
def test_parse(text, expected):
    assert parse(text) == expected


# --- Result ------------------------------------------------------------------

def test_result_score_labeled_and_correct():
    r = Result.score(make_row("q1", "vidA", answer_idx=2), "vidA", " 2\n")
    assert (r.pred, r.gold, r.correct, r.raw) == (2, 2, True, "2")
    assert r.video == "vidA" and r.qid == "q1"


def test_result_score_labeled_and_wrong():
    r = Result.score(make_row("q1", "vidA", answer_idx=2), "vidA", "1")
    assert (r.pred, r.gold, r.correct) == (1, 2, False)


def test_result_score_unparsed_counts_as_wrong_not_missing():
    r = Result.score(make_row("q1", "vidA", answer_idx=2), "vidA", "I cannot tell")
    assert r.pred is None
    assert r.correct is False


def test_result_score_unlabeled_is_not_scored():
    r = Result.score(make_row("q1", "vidA", answer_idx=None), "vidA", "2")
    assert r.pred == 2
    assert r.gold is None
    assert r.correct is None


def test_result_score_coerces_numpy_gold_to_int():
    """A DataFrame round-trip yields np.int64, which json.dumps refuses."""
    row = pd.DataFrame([make_row("q1", "vidA", answer_idx=2)]).to_dict(orient="records")[0]
    r = Result.score(row, "vidA", "2")
    assert type(r.gold) is int
    assert json.loads(json.dumps(r.as_dict()))["gold"] == 2


def test_result_score_treats_nan_gold_as_unlabeled():
    df = pd.DataFrame([make_row("q1", "vidA", answer_idx=1), make_row("q2", "vidA", answer_idx=None)])
    unlabeled = df.to_dict(orient="records")[1]
    r = Result.score(unlabeled, "vidA", "1")
    assert r.gold is None
    assert r.correct is None


def test_result_round_trips_through_json():
    r = Result.score(make_row("q1", "vidA", answer_idx=None), "vidA", "nope")
    assert Result(**json.loads(json.dumps(r.as_dict()))) == r


# --- group_by_video ----------------------------------------------------------

def test_group_by_video_preserves_order():
    rows = [make_row("q1", "vidA"), make_row("q2", "vidB"), make_row("q3", "vidA")]
    groups = group_by_video(rows)
    assert list(groups) == ["vidA", "vidB"]
    assert [r[Columns.QID] for r in groups["vidA"]] == ["q1", "q3"]


def test_group_by_video_accepts_dataframe():
    df = pd.DataFrame([make_row("q1", "vidA"), make_row("q2", "vidA")])
    assert len(group_by_video(df)["vidA"]) == 2


# --- summarize ---------------------------------------------------------------

def _result(qid="q1", pred=0, gold=0, raw="0"):
    correct = None if gold is None else (pred is not None and pred == gold)
    return Result(video="vidA", qid=qid, pred=pred, gold=gold, correct=correct, raw=raw)


def test_summarize_empty_returns_none():
    assert summarize([]) is None


def test_summarize_accuracy():
    results = [_result("q1", 0, 0), _result("q2", 1, 0), _result("q3", 2, 2), _result("q4", 3, 3)]
    assert summarize(results) == pytest.approx(0.75)


def test_summarize_counts_unparsed_as_wrong():
    # 2 of 3 scored rows are right; the unparsed row is wrong, not absent.
    results = [_result("q1", 0, 0), _result("q2", 1, 1), _result("q3", None, 2, raw="hmm")]
    assert summarize(results) == pytest.approx(2 / 3)


def test_summarize_unlabeled_returns_none():
    assert summarize([_result("q1", 0, None), _result("q2", 1, None)]) is None


def test_summarize_accepts_jsonl_dicts():
    dicts = [_result("q1", 0, 0).as_dict(), _result("q2", 1, 0).as_dict()]
    assert summarize(dicts) == pytest.approx(0.5)


def test_summarize_rejects_off_schema_dicts():
    with pytest.raises(TypeError):
        summarize([{"qid": "q1", "pred": 0}])


# --- to_frame ----------------------------------------------------------------

def test_to_frame_keeps_missing_values_distinguishable():
    df = to_frame([_result("q1", 2, 2), _result("q2", None, None, raw="hmm")])
    assert list(df.columns) == ["video", "qid", "pred", "gold", "correct", "raw"]
    assert df.loc[0, "pred"] == 2 and df.loc[0, "correct"]
    assert df["pred"].isna().tolist() == [False, True]
    assert df["correct"].isna().tolist() == [False, True]


def test_to_frame_empty_has_columns():
    df = to_frame([])
    assert df.empty
    assert list(df.columns) == ["video", "qid", "pred", "gold", "correct", "raw"]


# --- run ---------------------------------------------------------------------

def test_run_decodes_each_video_once():
    rows = [make_row(f"q{i}", "vidA") for i in range(3)] + [make_row("q9", "vidB")]
    vlm = FakeVLM(max_batch_size=8)
    run(vlm, rows, {"vidA": "ta", "vidB": "tb"})
    assert vlm.prepared == ["vidA", "vidB"]


@pytest.mark.parametrize(
    "max_batch_size, expected",
    [(1, [1, 1, 1, 1, 1, 1]), (4, [4, 2]), (8, [6])],
)
def test_run_respects_max_batch_size(max_batch_size, expected):
    rows = [make_row(f"q{i}", "vidA") for i in range(6)]
    vlm = FakeVLM(max_batch_size=max_batch_size)
    run(vlm, rows, {"vidA": "t"})
    assert vlm.batch_sizes == expected


def test_run_scores_against_gold():
    rows = [
        make_row("q1", "vidA", answer_idx=2),
        make_row("q2", "vidA", answer_idx=1),
        make_row("q3", "vidA", answer_idx=0),
    ]
    vlm = FakeVLM(replies={"q1": "2", "q2": "Answer: 3", "q3": "I cannot tell"}, max_batch_size=8)
    results = run(vlm, rows, {"vidA": "t"})

    assert [r.pred for r in results] == [2, 3, None]
    assert [r.gold for r in results] == [2, 1, 0]
    assert [r.correct for r in results] == [True, False, False]
    assert [r.raw for r in results] == ["2", "Answer: 3", "I cannot tell"]
    assert all(r.video == "vidA" for r in results)


def test_run_skips_undecodable_video_and_continues():
    rows = [make_row("q1", "bad"), make_row("q2", "good"), make_row("q3", "good")]
    vlm = FakeVLM(max_batch_size=8, undecodable=("bad",))
    results = run(vlm, rows, {"bad": "t", "good": "t"})

    assert vlm.prepared == ["good"]
    assert [r.qid for r in results] == ["q2", "q3"]


def test_run_all_videos_undecodable_returns_empty():
    vlm = FakeVLM(undecodable=("bad",))
    assert run(vlm, [make_row("q1", "bad")], {"bad": "t"}) == []


def test_run_accepts_dataframe_rows():
    df = pd.DataFrame([make_row("q1", "vidA", answer_idx=1)])
    vlm = FakeVLM(replies={"q1": "1"}, max_batch_size=8)
    results = run(vlm, df, {"vidA": "t"})
    assert [r.correct for r in results] == [True]


def test_run_passes_transcript_through():
    vlm = FakeVLM(max_batch_size=8)
    run(vlm, [make_row("q1", "vidA")], {"vidA": "hello transcript"}, use_transcript=True)
    assert vlm.transcripts_seen == ["hello transcript"]
    assert vlm.use_transcript_seen == [True]


def test_run_forwards_use_transcript_false():
    vlm = FakeVLM(max_batch_size=8)
    run(vlm, [make_row("q1", "vidA")], {"vidA": "hello"}, use_transcript=False)
    assert vlm.use_transcript_seen == [False]


def test_run_missing_transcript_becomes_empty_string():
    vlm = FakeVLM(max_batch_size=8)
    run(vlm, [make_row("q1", "vidA")], {})  # no entry for vidA
    assert vlm.transcripts_seen == [""]


# --- persistence and resume --------------------------------------------------

def test_run_writes_jsonl(tmp_path):
    out = tmp_path / "results.jsonl"
    rows = [make_row("q1", "vidA", answer_idx=1), make_row("q2", "vidA", answer_idx=0)]
    vlm = FakeVLM(replies={"q1": "1", "q2": "3"}, max_batch_size=1)
    results = run(vlm, rows, {"vidA": "t"}, out_path=out)

    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert written == [r.as_dict() for r in results]
    assert [w["correct"] for w in written] == [True, False]


def test_run_writes_null_for_unparsed_and_unlabeled(tmp_path):
    out = tmp_path / "results.jsonl"
    vlm = FakeVLM(replies={"q1": "no idea"}, max_batch_size=8)
    run(vlm, [make_row("q1", "vidA", answer_idx=None)], {"vidA": "t"}, out_path=out)

    written = json.loads(out.read_text().splitlines()[0])
    assert written["pred"] is None and written["gold"] is None and written["correct"] is None


def test_run_resume_skips_completed_qids(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(json.dumps(_result("q1", 1, 1, raw="1").as_dict()) + "\n")

    rows = [make_row("q1", "vidA", answer_idx=1), make_row("q2", "vidA", answer_idx=0)]
    vlm = FakeVLM(max_batch_size=8)
    results = run(vlm, rows, {"vidA": "t"}, out_path=out, resume=True)

    assert vlm.asked == ["q2"]                  # q1 not re-generated
    assert [r.qid for r in results] == ["q2"]   # return covers this call only
    assert len(out.read_text().splitlines()) == 2  # file accumulates both


def test_run_resume_skips_fully_completed_video_without_decoding(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(json.dumps(_result("q1", 0, 0).as_dict()) + "\n")

    rows = [make_row("q1", "vidA"), make_row("q2", "vidB")]
    vlm = FakeVLM(max_batch_size=8)
    run(vlm, rows, {"vidA": "t", "vidB": "t"}, out_path=out, resume=True)
    assert vlm.prepared == ["vidB"]  # vidA never decoded


def test_run_resume_false_reruns_everything(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(json.dumps(_result("q1", 0, 0).as_dict()) + "\n")

    vlm = FakeVLM(max_batch_size=8)
    run(vlm, [make_row("q1", "vidA")], {"vidA": "t"}, out_path=out, resume=False)
    assert vlm.asked == ["q1"]


def test_summarize_reads_back_a_finished_run(tmp_path):
    """The JSONL is the artifact: scoring it later must not need the model."""
    out = tmp_path / "results.jsonl"
    rows = [make_row("q1", "vidA", answer_idx=1), make_row("q2", "vidA", answer_idx=0)]
    vlm = FakeVLM(replies={"q1": "1", "q2": "3"}, max_batch_size=8)
    run(vlm, rows, {"vidA": "t"}, out_path=out)

    reloaded = [json.loads(line) for line in out.read_text().splitlines()]
    assert summarize(reloaded) == pytest.approx(0.5)
