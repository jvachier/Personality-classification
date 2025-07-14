"""Simple test to verify pytest is working."""

import pytest


def test_simple():
    """Simple test that should always pass."""
    assert True


def test_basic_math():
    """Test basic math operations."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6
