import pytest


@pytest.fixture(autouse=True)
def _offline(no_network):
    """Integration tests must never reach a real provider."""
