"""Tests for ChronoBox CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from chronobox.cli.main import build_parser, main


class TestBuildParser:
    """Tests for argument parser construction."""

    def test_parser_creates_subcommands(self) -> None:
        parser = build_parser()
        assert parser is not None

    def test_version_flag(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_no_command_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0


class TestEstimateCommand:
    """Tests for the estimate command."""

    def test_estimate_arima_airline(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["estimate", "arima", "--data", "airline", "--order", "0,1,1",
              "--seasonal", "0,1,1,12", "--column", "passengers"])
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_estimate_unknown_model(self) -> None:
        with pytest.raises(SystemExit):
            main(["estimate", "unknown_model", "--data", "airline"])


class TestTestCommand:
    """Tests for the test command."""

    def test_adf_test(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["test", "adf", "--data", "airline", "--column", "passengers"])
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_unknown_test(self) -> None:
        with pytest.raises(SystemExit):
            main(["test", "unknown_test", "--data", "airline"])


class TestForecastCommand:
    """Tests for the forecast command."""

    def test_forecast_arima(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["forecast", "--model", "arima", "--data", "airline",
              "--order", "0,1,1", "--steps", "6", "--column", "passengers"])
        captured = capsys.readouterr()
        assert "Forecast" in captured.out or len(captured.out) > 0


class TestFilterCommand:
    """Tests for the filter command."""

    def test_hp_filter(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["filter", "hp", "--data", "airline", "--column", "passengers"])
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_unknown_filter(self) -> None:
        with pytest.raises(SystemExit):
            main(["filter", "unknown_filter", "--data", "airline"])


class TestEntryPoint:
    """Tests for the entry point."""

    def test_module_execution(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "chronobox.cli.main", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout
