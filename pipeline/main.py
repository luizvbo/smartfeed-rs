"""SmartFeed Python pipeline entry point.

Run from the repository root:

    python pipeline/main.py                 # run once
    python pipeline/main.py --loop 300      # loop every 300 seconds
    python pipeline/main.py --retrain       # force retrain
    python pipeline/main.py --dry-run       # do not write to DB

The pipeline:
  1. Ingests configured RSS/Atom feeds into ``news_items``.
  2. Computes multilingual sentence embeddings and caches them.
  3. Trains a lightweight preference model on the user's votes.
  4. Scores every news item and updates ``news_items.model_score``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running as `python pipeline/main.py` from the repo root by adding
# pipeline/ to sys.path so sibling modules (config, db, rss, embed, model) are
# importable.
PY_DIR = Path(__file__).resolve().parent
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

import db as dbmod
from config import Config, load_config, setup_logging
from embed import Embedder, compute_missing_embeddings
from model import score_all, should_retrain, train
from rss import ingest_all

log = logging.getLogger("smartfeed.pipeline")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="smartfeed-pipeline",
        description="Ingest RSS feeds, embed items, train a preference model, and score news items.",
    )
    p.add_argument(
        "--loop",
        type=int,
        default=0,
        metavar="SECONDS",
        help="run continuously, sleeping SECONDS between runs (0 = run once)",
    )
    p.add_argument(
        "--retrain", action="store_true", help="force a retrain of the preference model"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="do not write to the database (schema is still created)",
    )
    return p.parse_args(argv)


def run_once(
    cfg: Config, embedder: Embedder, force_retrain: bool, dry_run: bool
) -> None:
    conn = dbmod.connect(cfg)
    try:
        dbmod.init_schema(conn)
        if dry_run:
            log.info("dry-run mode: skipping feed ingestion and DB writes")
            return

        # 1. Ingest feeds.
        new_items = ingest_all(conn, cfg.feeds, cfg.max_age_days)
        log.info("ingestion complete: %d new item(s)", new_items)

        # 2. Compute missing embeddings.
        computed = compute_missing_embeddings(conn, embedder)
        log.info("embeddings: %d computed", computed)

        # 3. Train if needed.
        if should_retrain(conn, cfg, force=force_retrain):
            log.info("training preference model")
            train(conn, cfg)
        else:
            log.info("no new votes since last training; skipping retrain")

        # 4. Score every item.
        scored = score_all(conn, cfg, embedder)
        log.info("scoring complete: %d item(s) scored", scored)
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    setup_logging(cfg.log_level)
    log.info(
        "smartfeed pipeline starting (db=%s, scoring=%s, model=%s)",
        cfg.db_path,
        cfg.scoring_method,
        cfg.model_name,
    )

    embedder = Embedder(cfg)

    if args.loop and args.loop > 0:
        log.info("loop mode: running every %d seconds", args.loop)
        try:
            while True:
                start = time.time()
                try:
                    run_once(
                        cfg, embedder, force_retrain=args.retrain, dry_run=args.dry_run
                    )
                except Exception as exc:  # noqa: BLE001
                    log.error("run failed: %s", exc)
                elapsed = time.time() - start
                sleep_for = max(0, args.loop - elapsed)
                log.info("run finished in %.1fs; sleeping %.1fs", elapsed, sleep_for)
                time.sleep(sleep_for)
        except KeyboardInterrupt:
            log.info("interrupted; exiting")
            return 0
    else:
        run_once(cfg, embedder, force_retrain=args.retrain, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
