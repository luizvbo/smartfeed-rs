"""RSS/Atom feed ingestion.

Fetches feeds with httpx, parses them with feedparser, extracts clean text and
image URLs with BeautifulSoup, and inserts new items into ``news_items``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin

import feedparser
import httpx
from bs4 import BeautifulSoup

from config import USER_AGENT, FeedConfig
from db import (
    insert_news_item,
    is_feed_active,
    update_feed_status,
    upsert_feed,
    url_exists,
)

log = logging.getLogger(__name__)

MAX_SUMMARY_CHARS = 1000


@dataclass
class ParsedItem:
    title: str
    url: str
    summary: str | None
    image_url: str | None
    published_at: int | None


def _strip_html(html: str | None) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text[:MAX_SUMMARY_CHARS]


def _parse_published(entry: dict[str, Any]) -> int | None:
    # feedparser exposes several possible fields; prefer the structured one.
    for field in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed = entry.get(field)
        if parsed:
            try:
                dt = datetime(*parsed[:6], tzinfo=UTC)
                return int(dt.timestamp())
            except (TypeError, ValueError):
                continue
    # Fall back to a string field parsed by feedparser into a struct if present.
    for field in ("published", "updated", "created"):
        raw = entry.get(field)
        if isinstance(raw, str) and raw:
            parsed = feedparser._parse_date(raw)  # type: ignore[attr-defined]
            if parsed:
                try:
                    dt = datetime(*parsed[:6], tzinfo=UTC)
                    return int(dt.timestamp())
                except (TypeError, ValueError):
                    continue
    return None


def _extract_image_from_media(entry: dict[str, Any]) -> str | None:
    # 1. media_thumbnail
    thumb = entry.get("media_thumbnail")
    if isinstance(thumb, list) and thumb:
        first = thumb[0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]
    # 2. media_content
    content = entry.get("media_content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict) and first.get("url"):
            return first["url"]
    # 3. enclosures
    enclosures = entry.get("enclosures")
    if isinstance(enclosures, list):
        for enc in enclosures:
            if isinstance(enc, dict):
                href = enc.get("href") or enc.get("url")
                if href:
                    return href
    return None


def _extract_image_from_html(html: str | None, base_url: str | None) -> str | None:
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    img = soup.find("img")
    if img is None:
        return None
    src = img.get("src")
    if not src:
        return None
    if base_url:
        return urljoin(base_url, src)
    return src


def _extract_image(entry: dict[str, Any], feed_url: str) -> str | None:
    img = _extract_image_from_media(entry)
    if img:
        return img
    # Try content/summary HTML.
    base = entry.get("link") or feed_url
    for field in ("content", "summary", "description"):
        value = entry.get(field)
        if isinstance(value, list) and value:
            # feedparser content is a list of dicts with 'value'
            first = value[0]
            if isinstance(first, dict):
                img = _extract_image_from_html(first.get("value"), base)
                if img:
                    return img
        elif isinstance(value, str):
            img = _extract_image_from_html(value, base)
            if img:
                return img
    return None


def _entry_url(entry: dict[str, Any]) -> str | None:
    return entry.get("link") or entry.get("id")


def _parse_entries(parsed: Any, feed_url: str) -> list[ParsedItem]:
    items: list[ParsedItem] = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        url = _entry_url(entry)
        if not title or not url:
            continue
        summary_raw = entry.get("summary") or entry.get("description")
        if isinstance(summary_raw, list) and summary_raw:
            summary_raw = (
                summary_raw[0].get("value")
                if isinstance(summary_raw[0], dict)
                else None
            )
        summary = _strip_html(summary_raw) or None
        image_url = _extract_image(entry, feed_url)
        published_at = _parse_published(entry)
        items.append(
            ParsedItem(
                title=title,
                url=url,
                summary=summary,
                image_url=image_url,
                published_at=published_at,
            )
        )
    return items


def fetch_feed(feed: FeedConfig, timeout: float = 20.0) -> tuple[Any, str]:
    """Fetch raw feed bytes and parse with feedparser. Returns (parsed, raw_url)."""
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=timeout,
    ) as client:
        resp = client.get(feed.url)
        resp.raise_for_status()
        body = resp.content
    # Let feedparser handle the decoding using the response content.
    parsed = feedparser.parse(body)
    return parsed, feed.url


def ingest_feed(conn, feed: FeedConfig, max_age_days: int) -> int:
    """Ingest a single feed. Returns the number of new items inserted."""
    upsert_feed(conn, feed.url, feed.title)
    if not is_feed_active(conn, feed.url):
        log.info("feed %s is disabled in the database; skipping", feed.url)
        return 0
    try:
        parsed, _ = fetch_feed(feed)
    except Exception as exc:  # noqa: BLE001 - any fetch/parse failure is recoverable
        log.error("failed to fetch feed %s: %s", feed.url, exc)
        update_feed_status(conn, feed.url, "error", str(exc)[:500])
        return 0

    if parsed.bozo and not parsed.entries:
        log.warning(
            "feed %s parsed with errors: %s",
            feed.url,
            getattr(parsed, "bozo_exception", ""),
        )
        update_feed_status(
            conn,
            feed.url,
            "error",
            str(getattr(parsed, "bozo_exception", "parse error"))[:500],
        )
        return 0

    cutoff = int(time.time()) - max_age_days * 86400
    new_count = 0
    for item in _parse_entries(parsed, feed.url):
        if item.published_at is not None and item.published_at < cutoff:
            continue
        if url_exists(conn, item.url):
            continue
        inserted = insert_news_item(
            conn,
            title=item.title,
            url=item.url,
            source_feed=feed.url,
            summary=item.summary,
            image_url=item.image_url,
            published_at=item.published_at,
        )
        if inserted is not None:
            new_count += 1
            log.info("new item id=%s: %s", inserted, item.title[:80])

    update_feed_status(conn, feed.url, "ok", None)
    log.info("feed %s: %d new item(s)", feed.url, new_count)
    return new_count


def ingest_all(conn, feeds: list[FeedConfig], max_age_days: int) -> int:
    total = 0
    for feed in feeds:
        try:
            total += ingest_feed(conn, feed, max_age_days)
        except Exception as exc:  # noqa: BLE001
            log.error("unexpected error ingesting feed %s: %s", feed.url, exc)
            try:
                update_feed_status(conn, feed.url, "error", str(exc)[:500])
            except Exception:
                pass
    return total
