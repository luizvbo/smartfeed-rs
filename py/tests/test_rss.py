from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import db as dbmod
import rss
from config import FeedConfig


class TestStripHtml:
    def test_removes_tags_and_truncates(self):
        html = "<p>Hello <b>world</b></p>" + "x" * 2000
        text = rss._strip_html(html)
        assert "<" not in text
        assert text.startswith("Hello world")
        assert len(text) <= rss.MAX_SUMMARY_CHARS

    def test_empty(self):
        assert rss._strip_html(None) == ""
        assert rss._strip_html("") == ""


class TestParsePublished:
    def test_published_parsed_tuple(self):
        entry = {"published_parsed": (2024, 1, 1, 12, 0, 0)}
        ts = rss._parse_published(entry)
        expected = int(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        assert ts == expected

    def test_string_parsed(self, monkeypatch):
        entry = {"published": "Mon, 01 Jan 2024 12:00:00 GMT"}
        monkeypatch.setattr(
            rss.feedparser, "_parse_date", lambda s: (2024, 1, 1, 12, 0, 0), raising=False
        )
        ts = rss._parse_published(entry)
        expected = int(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        assert ts == expected

    def test_missing_returns_none(self):
        assert rss._parse_published({}) is None


class TestExtractImage:
    def test_media_thumbnail(self):
        entry = {"media_thumbnail": [{"url": "https://example.com/thumb.jpg"}]}
        assert rss._extract_image(entry, "https://example.com/feed") == "https://example.com/thumb.jpg"

    def test_enclosure_href(self):
        entry = {"enclosures": [{"href": "https://example.com/img.jpg"}]}
        assert rss._extract_image(entry, "https://example.com/feed") == "https://example.com/img.jpg"

    def test_html_img_relative(self):
        entry = {"summary": '<p><img src="/pic.png" /></p>', "link": "https://example.com/"}
        assert rss._extract_image(entry, "https://example.com/feed") == "https://example.com/pic.png"

    def test_no_image(self):
        assert rss._extract_image({}, "https://example.com/feed") is None


class TestParseEntries:
    def test_basic(self):
        parsed = SimpleNamespace(
            entries=[
                {
                    "title": "  Title  ",
                    "link": "https://example.com/article",
                    "summary": "<p>Body</p>",
                    "published_parsed": (2024, 1, 1, 0, 0, 0),
                }
            ]
        )
        items = rss._parse_entries(parsed, "https://example.com/feed")
        assert len(items) == 1
        assert items[0].title == "Title"
        assert items[0].url == "https://example.com/article"
        assert items[0].summary == "Body"

    def test_skips_invalid(self):
        parsed = SimpleNamespace(entries=[{"title": "Only title"}])
        items = rss._parse_entries(parsed, "https://example.com/feed")
        assert items == []


class TestIngestFeed:
    def _make_good_fetch(self, feed):
        return SimpleNamespace(
            bozo=False,
            entries=[
                {
                    "title": "  New Article  ",
                    "link": "https://example.com/new",
                    "summary": "<p>Summary</p>",
                }
            ],
        ), feed.url

    def test_inserts_new_items(self, conn, monkeypatch):
        monkeypatch.setattr(rss, "fetch_feed", self._make_good_fetch)
        feed = FeedConfig(url="https://example.com/feed.xml", title="Example")
        count = rss.ingest_feed(conn, feed, max_age_days=365)
        assert count == 1
        assert dbmod.url_exists(conn, "https://example.com/new")

    def test_does_not_duplicate(self, conn, monkeypatch):
        dbmod.upsert_feed(conn, "https://example.com/feed.xml", "Example")
        dbmod.insert_news_item(
            conn,
            title="Existing",
            url="https://example.com/existing",
            source_feed="https://example.com/feed.xml",
            summary="s",
            image_url=None,
            published_at=1,
        )

        def fetch(feed):
            return SimpleNamespace(
                bozo=False,
                entries=[
                    {
                        "title": "Existing",
                        "link": "https://example.com/existing",
                        "summary": "s",
                    }
                ],
            ), feed.url

        monkeypatch.setattr(rss, "fetch_feed", fetch)
        feed = FeedConfig(url="https://example.com/feed.xml", title="Example")
        count = rss.ingest_feed(conn, feed, max_age_days=365)
        assert count == 0

    def test_handles_fetch_failure(self, conn, monkeypatch):
        def bad_fetch(feed):
            raise ConnectionError("nope")

        monkeypatch.setattr(rss, "fetch_feed", bad_fetch)
        feed = FeedConfig(url="https://example.com/bad.xml", title="Bad")
        count = rss.ingest_feed(conn, feed, max_age_days=365)
        assert count == 0
        row = conn.execute("SELECT last_status, error_message FROM feeds WHERE url=?", (feed.url,)).fetchone()
        assert row["last_status"] == "error"

    def test_bozo_feed_updates_status(self, conn, monkeypatch):
        def bozo_fetch(feed):
            return SimpleNamespace(
                bozo=True,
                entries=[],
                bozo_exception="malformed",
            ), feed.url

        monkeypatch.setattr(rss, "fetch_feed", bozo_fetch)
        feed = FeedConfig(url="https://example.com/bozo.xml", title="Bozo")
        count = rss.ingest_feed(conn, feed, max_age_days=365)
        assert count == 0
        row = conn.execute("SELECT last_status, error_message FROM feeds WHERE url=?", (feed.url,)).fetchone()
        assert row["last_status"] == "error"

    def test_skips_old_entries(self, conn, monkeypatch):
        def old_fetch(feed):
            return SimpleNamespace(
                bozo=False,
                entries=[
                    {
                        "title": "Old",
                        "link": "https://example.com/old",
                        "summary": "x",
                        "published_parsed": (2020, 1, 1, 0, 0, 0),
                    }
                ],
            ), feed.url

        monkeypatch.setattr(rss, "fetch_feed", old_fetch)
        feed = FeedConfig(url="https://example.com/old.xml", title="Old")
        count = rss.ingest_feed(conn, feed, max_age_days=1)
        assert count == 0


class TestIngestAll:
    def test_sums_feeds(self, conn, monkeypatch):
        calls = []

        def fake_ingest_feed(c, f, days):
            calls.append(f)
            return 1

        monkeypatch.setattr(rss, "ingest_feed", fake_ingest_feed)
        feeds = [
            FeedConfig(url="https://example.com/a.xml"),
            FeedConfig(url="https://example.com/b.xml"),
        ]
        total = rss.ingest_all(conn, feeds, max_age_days=365)
        assert total == 2
        assert len(calls) == 2

    def test_continues_on_error(self, conn, monkeypatch):
        def fake_ingest_feed(c, f, days):
            if "bad" in f.url:
                raise ValueError("boom")
            return 1

        monkeypatch.setattr(rss, "ingest_feed", fake_ingest_feed)
        feeds = [
            FeedConfig(url="https://example.com/good.xml"),
            FeedConfig(url="https://example.com/bad.xml"),
            FeedConfig(url="https://example.com/good2.xml"),
        ]
        total = rss.ingest_all(conn, feeds, max_age_days=365)
        assert total == 2


class TestParsePublishedMore:
    def test_updated_parsed(self):
        entry = {"updated_parsed": (2024, 6, 15, 10, 0, 0)}
        ts = rss._parse_published(entry)
        assert ts == int(datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc).timestamp())

    def test_created_parsed(self):
        entry = {"created_parsed": (2024, 3, 1, 8, 0, 0)}
        ts = rss._parse_published(entry)
        assert ts == int(datetime(2024, 3, 1, 8, 0, 0, tzinfo=timezone.utc).timestamp())

    def test_bad_published_parsed_then_string(self, monkeypatch):
        entry = {"published_parsed": "not-a-tuple", "published": "Mon, 01 Jan 2024 12:00:00 GMT"}
        monkeypatch.setattr(
            rss.feedparser, "_parse_date", lambda s: (2024, 1, 1, 12, 0, 0), raising=False
        )
        ts = rss._parse_published(entry)
        expected = int(datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        assert ts == expected


class TestExtractImageMore:
    def test_media_content(self):
        entry = {"media_content": [{"url": "https://example.com/content.jpg"}]}
        assert rss._extract_image(entry, "https://example.com/feed") == "https://example.com/content.jpg"

    def test_enclosure_url(self):
        entry = {"enclosures": [{"url": "https://example.com/enc.jpg"}]}
        assert rss._extract_image(entry, "https://example.com/feed") == "https://example.com/enc.jpg"

    def test_content_list(self):
        entry = {
            "content": [{"value": '<p><img src="https://example.com/in-content.jpg" /></p>'}],
            "link": "https://example.com/",
        }
        assert rss._extract_image(entry, "https://example.com/feed") == "https://example.com/in-content.jpg"

    def test_description_string(self):
        entry = {
            "description": '<p><img src="/desc.jpg" /></p>',
            "link": "https://example.com/",
        }
        assert rss._extract_image(entry, "https://example.com") == "https://example.com/desc.jpg"

    def test_img_without_src(self):
        entry = {"summary": "<p><img alt='x' /></p>", "link": "https://example.com/"}
        assert rss._extract_image(entry, "https://example.com/feed") is None

    def test_img_without_base_url(self):
        entry = {"summary": '<p><img src="https://example.com/abs.jpg" /></p>'}
        assert rss._extract_image(entry, "https://example.com/feed") == "https://example.com/abs.jpg"


class TestParseEntriesMore:
    def test_summary_as_list(self):
        parsed = SimpleNamespace(
            entries=[
                {
                    "title": "List Summary",
                    "link": "https://example.com/article",
                    "summary": [{"value": "<p>Body</p>"}],
                }
            ]
        )
        items = rss._parse_entries(parsed, "https://example.com/feed")
        assert len(items) == 1
        assert items[0].summary == "Body"

    def test_url_from_id(self):
        parsed = SimpleNamespace(
            entries=[
                {
                    "title": "ID Only",
                    "id": "https://example.com/by-id",
                    "summary": "text",
                }
            ]
        )
        items = rss._parse_entries(parsed, "https://example.com/feed")
        assert len(items) == 1
        assert items[0].url == "https://example.com/by-id"
