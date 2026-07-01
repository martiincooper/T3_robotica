"""Thin wrapper around the CoppeliaSim ZeroMQ Remote API client.

Requires the ``coppeliasim_zmqremoteapi_client`` package that ships with
CoppeliaSim 4.4+ (``pip install coppeliasim-zmqremoteapi-client``) and a
running CoppeliaSim instance with the default ZMQ remote API server
enabled (it is on by default on port 23000).
"""
from __future__ import annotations


def connect(host: str = "localhost", port: int = 23000):
    """Return ``(client, sim)`` connected to a running CoppeliaSim.

    Raises a clear, actionable error if the package is missing or no
    instance is reachable.
    """
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Missing package 'coppeliasim_zmqremoteapi_client'.\n"
            "Install it with:  pip install coppeliasim-zmqremoteapi-client\n"
            "(it ships with CoppeliaSim 4.4+ under "
            "programming/zmqRemoteApi/clients/python)."
        ) from exc

    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    return client, sim


def version_string(sim) -> str:
    try:
        v = sim.getInt32Param(sim.intparam_program_version)
        rev = sim.getInt32Param(sim.intparam_program_revision)
        major, rest = divmod(v, 10000)
        minor, patch = divmod(rest, 100)
        return f"CoppeliaSim {major}.{minor}.{patch} (rev {rev})"
    except Exception as exc:  # pragma: no cover
        return f"<unknown version: {exc}>"
