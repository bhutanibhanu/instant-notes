import json
import time

from instant_notes.timing import Event, Tracer, compute_stats, load_records


def test_session_writes_jsonl_with_stop_to_text(tmp_path):
    """Drive a fake dictation session through start → marks → finalize and assert
    the JSONL record contains stop_to_text_ms."""
    path = tmp_path / "timing.jsonl"
    tr = Tracer(path=path)

    sid = tr.start()
    tr.mark(sid, Event.STOP)
    time.sleep(0.003)
    tr.mark(sid, Event.STT_REQUEST)
    time.sleep(0.003)
    tr.mark(sid, Event.STT_DONE)
    time.sleep(0.003)
    tr.mark(sid, Event.DISPLAYED)
    tr.mark(sid, Event.CLEANUP_START)
    tr.mark(sid, Event.CLEANUP_DONE)

    record = tr.finalize(sid)

    assert record is not None
    assert record["sid"] == sid
    assert record["stop_to_text_ms"] is not None and record["stop_to_text_ms"] >= 0
    assert record["tail_stt_ms"] is not None
    assert record["cleanup_ms"] is not None

    # JSONL line was written and round-trips with stop_to_text_ms present.
    parsed = json.loads(path.read_text().strip())
    assert parsed["sid"] == sid
    assert "stop_to_text_ms" in parsed


def test_stop_to_text_falls_back_to_stt_done(tmp_path):
    tr = Tracer(path=tmp_path / "t.jsonl")
    sid = tr.start()
    tr.mark(sid, Event.STOP)
    time.sleep(0.002)
    tr.mark(sid, Event.STT_DONE)  # no DISPLAYED → fall back to STT_DONE
    record = tr.finalize(sid)
    assert record["stop_to_text_ms"] is not None


def test_stats_p50_p95_max(tmp_path):
    """`stats` aggregates p50/p95/max per metric over the JSONL log."""
    path = tmp_path / "timing.jsonl"
    rows = [{"stop_to_text_ms": float(v), "tail_stt_ms": 50.0} for v in range(1, 101)]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    records = load_records(path)
    assert len(records) == 100
    stats = compute_stats(records)

    s2t = stats["stop_to_text_ms"]
    assert s2t["n"] == 100
    assert s2t["p50"] == 50.5          # midpoint of 1..100
    assert abs(s2t["p95"] - 95.05) < 0.05
    assert s2t["max"] == 100.0
    assert stats["tail_stt_ms"]["p50"] == 50.0
    # missing metrics simply absent
    assert "cleanup_ms" not in stats


def test_load_records_missing_file(tmp_path):
    assert load_records(tmp_path / "nope.jsonl") == []


def test_marks_never_raise_and_unknown_sid_is_noop(tmp_path):
    tr = Tracer(path=tmp_path / "t.jsonl")
    tr.mark(None, Event.STOP)          # no session → no-op, no raise
    tr.mark("nonexistent", Event.STOP)  # unknown sid → no-op
    assert tr.finalize(None) is None
    assert tr.finalize("nonexistent") is None
    assert not (tmp_path / "t.jsonl").exists()  # nothing written for no-ops
