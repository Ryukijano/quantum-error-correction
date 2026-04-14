"""Tests for registry-driven decoder discovery in services."""

from __future__ import annotations

from app.services import circuit_services
from syndrome_net import get_container


def test_service_decoder_names_follow_container_registry() -> None:
    names = circuit_services.get_decoder_names()
    assert isinstance(names, list)
    assert "mwpm" in names
    assert "union_find" in names


def test_build_threshold_decoder_uses_container_lookup():
    decoder = circuit_services.build_threshold_decoder("mwpm")
    assert decoder.name == "mwpm"

    unknown = "does_not_exist"
    try:
        circuit_services.build_threshold_decoder(unknown)
    except KeyError as exc:
        assert unknown in str(exc)
    else:
        raise AssertionError("Expected KeyError for unknown decoder")


def test_build_threshold_decoder_resolves_ising_backend():
    decoder = circuit_services.build_threshold_decoder("ising")
    assert decoder.name == "ising"


def test_container_registers_new_decoder_backends():
    container = get_container()
    decoders = set(container.decoders.list())
    assert "cudaq" in decoders
    assert "qujax" in decoders
    assert "cuqnn" in decoders
