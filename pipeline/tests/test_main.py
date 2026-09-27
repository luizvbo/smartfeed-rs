from __future__ import annotations

import main


class TestParseArgs:
    def test_defaults(self):
        args = main.parse_args([])
        assert args.loop == 0
        assert args.retrain is False
        assert args.dry_run is False

    def test_loop(self):
        args = main.parse_args(["--loop", "300"])
        assert args.loop == 300

    def test_retrain(self):
        args = main.parse_args(["--retrain"])
        assert args.retrain is True

    def test_dry_run(self):
        args = main.parse_args(["--dry-run"])
        assert args.dry_run is True
