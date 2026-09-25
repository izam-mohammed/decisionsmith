import socket

import pytest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Integration tests must never reach a real provider: every connection outside this machine fails."""
    real = socket.socket.connect

    def connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost") and not str(host).startswith("/"):
            raise OSError("network is blocked in integration tests (tried %r)" % (address,))
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
