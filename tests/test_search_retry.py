"""search 重试兜底：网络错误不再炸整跑，穷尽后返回空列表"""
import httpx

import config
import tools.search as search_mod


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeHttpx:
    """前 n 次抛连接错误，之后成功；记录调用次数与退避 sleep"""

    HTTPError = httpx.HTTPError  # search.py 的 except httpx.HTTPError 引用

    def __init__(self, failures, payload):
        self.failures = failures
        self.payload = payload
        self.calls = 0
        self.sleeps = []

    def post(self, *a, **kw):
        self.calls += 1
        if self.calls <= self.failures:
            raise httpx.ConnectError("conn refused")
        return _Resp(self.payload)

    def sleep(self, s):
        self.sleeps.append(s)


def _run(monkeypatch, failures, payload=None):
    fake = _FakeHttpx(failures, payload or {"results": [{"title": "t", "url": "http://e/1", "content": "c"}]})
    monkeypatch.setattr(config, "TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(config, "SEARCH_RETRIES", 3)
    monkeypatch.setattr(search_mod, "httpx", fake)
    monkeypatch.setattr(search_mod, "time", fake)  # time.sleep → fake.sleep 记录退避
    return search_mod.search("查询"), fake


def test_retry_then_success(monkeypatch):
    results, fake = _run(monkeypatch, failures=2)
    assert fake.calls == 3
    assert results and results[0]["url"] == "http://e/1"
    assert fake.sleeps == [1, 2]  # 1s, 2s 退避


def test_retry_exhausted_returns_empty(monkeypatch):
    results, fake = _run(monkeypatch, failures=99)
    assert results == []
    assert fake.calls == 3  # SEARCH_RETRIES=3 次后放弃，不抛异常


def test_no_key_returns_empty(monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "")
    assert search_mod.search("查询") == []
