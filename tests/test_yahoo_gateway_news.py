"""Ticker news from Yahoo's content gateway, which replaced the quote feed yfinance calls."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import tradingagents.dataflows.vendors.yahoo.news as ynews


def _story(title, when, publisher="Reuters"):
    return {"asset": {"title": title, "contentAttributes": {
        "summary": f"about {title}", "publishTime": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "canonicalUrl": f"https://finance.yahoo.com/{title}", "provider": {"displayName": publisher}}}}


def _gateway(monkeypatch, stories, page_size=50):
    """A gateway serving ``stories`` (newest first) in pages; records each start offset."""
    starts = []

    def page(canonical, start):
        starts.append((canonical, start))
        chunk = stories[start:start + page_size]
        return {"assets": chunk, "pageInfo": {"hasNextPage": start + page_size < len(stories)}}

    monkeypatch.setattr(ynews, "_gateway_page", page)
    return starts


def _no_quote_feed(monkeypatch):
    class Unused:
        def __init__(self, *a, **k):
            raise AssertionError("the quote feed should not be asked")

    monkeypatch.setattr(ynews.yf, "Ticker", Unused)
    monkeypatch.setattr(ynews.yf, "Search", Unused)


@pytest.mark.unit
def test_gateway_news_in_the_window_is_reported(monkeypatch):
    _no_quote_feed(monkeypatch)
    _gateway(monkeypatch, [
        _story("AFTER", datetime(2026, 10, 16, 9, tzinfo=UTC)),
        _story("INSIDE", datetime(2026, 10, 15, 14, tzinfo=UTC), "Bloomberg"),
        _story("BEFORE", datetime(2026, 10, 7, 14, tzinfo=UTC)),
    ])

    out = ynews.get_news_yfinance("NVDA", "2026-10-08", "2026-10-15")

    assert "### INSIDE (source: Bloomberg)" in out
    assert "about INSIDE" in out and "Link: https://finance.yahoo.com/INSIDE" in out
    assert "AFTER" not in out and "BEFORE" not in out
    assert "Partial coverage" not in out  # the list reaches back past the window's start


@pytest.mark.unit
def test_a_window_the_list_only_partly_reaches_says_so(monkeypatch):
    _no_quote_feed(monkeypatch)
    _gateway(monkeypatch, [
        _story("LATE", datetime(2026, 10, 15, 20, tzinfo=UTC)),
        _story("EARLIER", datetime(2026, 10, 15, 10, 45, tzinfo=UTC)),
    ])

    out = ynews.get_news_yfinance("NVDA", "2026-10-08", "2026-10-15")

    assert "### LATE" in out and "### EARLIER" in out
    assert "Partial coverage" in out and "reaching back to 2026-10-15 10:45 UTC" in out
    assert "not an absence of news for NVDA" in out


@pytest.mark.unit
def test_gateway_pages_until_it_runs_out_or_hits_its_cap(monkeypatch):
    now = datetime.now(UTC)
    stories = [_story(f"S{i}", now - timedelta(minutes=i)) for i in range(260)]
    starts = _gateway(monkeypatch, stories)

    articles = ynews._gateway_news("NVDA")

    assert len(articles) == ynews.GATEWAY_MAX_ITEMS
    assert [s for _, s in starts] == [0, 50, 100, 150]
    assert articles[0]["title"] == "S0" and articles[0]["pub_date"].tzinfo is not None

    starts.clear()
    _gateway(monkeypatch, stories[:70])
    assert len(ynews._gateway_news("NVDA")) == 70


@pytest.mark.unit
def test_a_full_window_is_cut_to_the_limit_spread_over_its_time_range(monkeypatch):
    _no_quote_feed(monkeypatch)
    start = datetime(2026, 10, 15, 23, tzinfo=UTC)
    _gateway(monkeypatch, [_story(f"S{i:03d}", start - timedelta(minutes=10 * i)) for i in range(120)])

    out = ynews.get_news_yfinance("NVDA", "2026-10-14", "2026-10-15")

    titles = [line[4:8] for line in out.splitlines() if line.startswith("### ")]
    assert len(titles) == ynews.get_config()["news_article_limit"]
    assert titles[0] == "S000" and titles[-1] == "S119"  # newest and oldest both kept


@pytest.mark.unit
def test_a_window_older_than_the_gateway_reaches_is_unavailable(monkeypatch):
    _no_quote_feed(monkeypatch)
    _gateway(monkeypatch, [_story("RECENT", datetime(2026, 10, 15, 12, tzinfo=UTC))])

    out = ynews.get_news_yfinance("NVDA", "2026-09-01", "2026-09-08")

    assert "unavailable" in out and "not an absence" in out


@pytest.mark.unit
def test_a_failing_gateway_falls_back_to_the_quote_feed(monkeypatch):
    def broken(canonical, start):
        raise OSError("gateway down")

    monkeypatch.setattr(ynews, "_gateway_page", broken)
    today = datetime.now(UTC)

    class FakeTicker:
        def __init__(self, *a, **k):
            pass

        def get_news(self, count=20):
            return [{"title": "FROM FEED", "publisher": "P", "link": "l",
                     "providerPublishTime": int(today.timestamp())}]

    monkeypatch.setattr(ynews.yf, "Ticker", FakeTicker)
    day = today.strftime("%Y-%m-%d")

    assert "FROM FEED" in ynews.get_news_yfinance("NVDA", day, day)


@pytest.mark.unit
@pytest.mark.parametrize("n, limit, expect", [
    (5, 10, [0, 1, 2, 3, 4]),
    (10, 4, [0, 3, 6, 9]),
    (10, 1, [0]),
])
def test_spread_keeps_the_ends_and_spaces_the_rest(n, limit, expect):
    items = [{"i": i} for i in range(n)]
    assert [a["i"] for a in ynews._spread(items, limit)] == expect
