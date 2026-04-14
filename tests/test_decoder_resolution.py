"""Tests for centralized decoder resolution helpers."""
from __future__ import annotations

import pytest

from surface_code_in_stem.decoders.resolution import resolve_baseline_decoder, resolve_threshold_decoder


def test_resolve_baseline_decoder_uses_requested_decoder(monkeypatch):
    class FakeDecoder:
        name = "custom"

    class FakeContainer:
        def get_decoder(self, name: str):
            assert name == "custom"
            return FakeDecoder()

    monkeypatch.setattr("syndrome_net.container.get_container", lambda: FakeContainer())

    resolution = resolve_baseline_decoder("custom")
    assert resolution.requested_name == "custom"
    assert resolution.resolved_name == "custom"
    assert resolution.fallback_reason is None
    assert resolution.decoder.name == "custom"


def test_resolve_baseline_decoder_falls_back_to_local_mwpm(monkeypatch):
    class FakeContainer:
        def get_decoder(self, name: str):
            raise KeyError(name)

    monkeypatch.setattr("syndrome_net.container.get_container", lambda: FakeContainer())

    resolution = resolve_baseline_decoder("missing")
    assert resolution.requested_name == "missing"
    assert resolution.resolved_name == "mwpm"
    assert resolution.fallback_reason == "fallback_to_mwpm_decoder:missing"
    assert resolution.decoder.name == "mwpm"


def test_resolve_baseline_decoder_defaults_to_local_mwpm_when_none():
    resolution = resolve_baseline_decoder(None)
    assert resolution.requested_name is None
    assert resolution.resolved_name == "mwpm"
    assert resolution.fallback_reason is None
    assert resolution.decoder.name == "mwpm"


def test_resolve_threshold_decoder_rejects_unknown(monkeypatch):
    class FakeDecoders:
        def list(self):
            return ["mwpm"]

    class FakeContainer:
        def __init__(self):
            self.decoders = FakeDecoders()

        def get_decoder(self, name: str):
            raise AssertionError("get_decoder should not be called for unknown decoder")

    monkeypatch.setattr("syndrome_net.container.get_container", lambda: FakeContainer())

    with pytest.raises(KeyError, match="does_not_exist"):
        resolve_threshold_decoder("does_not_exist")
