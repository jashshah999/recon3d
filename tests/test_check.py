"""Tests for the recon3d check command."""

import pytest
from click.testing import CliRunner
from recon3d.cli import main


def test_check_runs():
    runner = CliRunner()
    result = runner.invoke(main, ["check"])
    assert result.exit_code == 0
    assert "Python:" in result.output
    assert "PyTorch:" in result.output
    assert "Backends:" in result.output


def test_check_lists_backends():
    runner = CliRunner()
    result = runner.invoke(main, ["check"])
    assert "VGGT:" in result.output
    assert "gsplat:" in result.output
    assert "GTSAM:" in result.output
    assert "Open3D:" in result.output
