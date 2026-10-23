"""URL 去重：Writer 引用编号与 Editor 参考来源共用同一列表"""
from state import unique_evidence


def test_dedup_keeps_first_occurrence_and_order():
    ev = [
        {"question": "q1", "title": "a", "url": "http://x/1", "summary": "s"},
        {"question": "q1", "title": "b", "url": "http://x/1", "summary": "s"},
        {"question": "q2", "title": "c", "url": "http://x/2", "summary": "s"},
        {"question": "q2", "title": "d", "url": "http://x/1", "summary": "s"},
    ]
    out = unique_evidence(ev)
    assert [e["url"] for e in out] == ["http://x/1", "http://x/2"]
    assert out[0]["title"] == "a"  # 保留首次出现


def test_dedup_empty_and_none():
    assert unique_evidence([]) == []
    assert unique_evidence(None) == []
    # 无 URL 的卡片无法判重，原样保留
    assert len(unique_evidence([{"title": "x", "url": "", "summary": ""}, {"title": "y", "url": "", "summary": ""}])) == 2


def test_editor_refs_deduped(fake_llm):
    """多轮累积的重复 URL 在成稿入口收敛（历史 bug：45 条参考同一 URL 重复 3 次）"""
    from agents.editor import editor
    dup = [{"question": "q", "title": f"t{i}", "url": "http://x/same", "summary": f"s{i}"} for i in range(4)]
    state = {"topic": "测试", "draft": "初稿", "final_evidence": dup}
    out = editor(state)
    assert out["report"].count("http://x/same") == 1
    assert out["trace"][-1]["dupes_removed"] == 3


def test_writer_deduped(fake_llm):
    from agents.writer import writer
    dup = [{"question": "q", "title": f"t{i}", "url": f"http://x/{i % 2}", "summary": "s"} for i in range(4)]
    out = writer({"topic": "测试", "final_evidence": dup})
    assert out["trace"][-1]["evidence_used"] == 2
    assert out["trace"][-1]["dupes_removed"] == 2
