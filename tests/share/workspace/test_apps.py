"""Focused tests for the current MP-Net workspace helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from share.workspace.mpnet import (
    _decode_mpnet,
    _encode_mpnet,
    add_primitive,
    attach_policy,
    create_template_mpnet,
    describe_transitions,
    list_primitives,
    set_terminal,
    summarize_mpnet,
)


@dataclass
class _FakePolicy:
    pretrained_path: Path | None = None
    device: str = "cpu"


def test_mpnet_helpers_encode_and_decode_template_config():
    """Template configs should round-trip through the workspace JSON helpers."""
    config = create_template_mpnet(notes="demo template")
    payload = _encode_mpnet(config)
    loaded = _decode_mpnet(payload)
    summary = summarize_mpnet(loaded)

    assert payload["start_primitive"] == "main"
    assert summary["primitive_count"] == 1
    assert summary["primitives"][0]["notes"] == "demo template"


def test_mpnet_helpers_attach_policies_and_describe_transitions(monkeypatch):
    """Policy attachment and transition summaries should match the edited config."""
    def fake_from_pretrained(policy_path: str, local_files_only: bool = False):
        assert policy_path == "/tmp/policy"
        assert local_files_only is True
        return _FakePolicy()

    monkeypatch.setattr("share.workspace.mpnet.PreTrainedConfig.from_pretrained", fake_from_pretrained)

    config = create_template_mpnet()
    add_primitive(config, "finish", template_from="main", is_terminal=True, connect_from="main")
    set_terminal(config, "main", False)
    attach_policy(config, "main", "/tmp/policy", {"steps": 42})

    primitives = {primitive["name"]: primitive for primitive in list_primitives(config)}
    transitions = describe_transitions(config)

    assert primitives["main"]["policy_path"] == "/tmp/policy"
    assert primitives["main"]["policy_overwrites"] == {"steps": 42}
    assert primitives["finish"]["is_terminal"] is True
    assert len(transitions) == 1
    assert transitions[0]["type"] == "always"
    assert transitions[0]["source"] == "main"
    assert transitions[0]["target"] == "finish"
