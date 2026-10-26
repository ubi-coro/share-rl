"""rename_map lives on RecordConfig, not on the dataset config: it applies to the policies, with or without a dataset."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from share.configs.record import DatasetRecordConfig, RecordConfig
from share.utils import control_utils

RENAME = {"observation.images.cam_top": "observation.images.image2"}


@dataclasses.dataclass
class _PolicyConfig:
    pretrained_path: str | None = None
    device: str = "cpu"


class _Policy:
    def eval(self):
        return self


@pytest.fixture
def captured(monkeypatch):
    """Stub the policy loading; record what make_policy and make_pre_post_processors were called with."""
    calls = SimpleNamespace(make_policy=[], processors=[])
    monkeypatch.setattr(control_utils, "PreTrainedConfig", SimpleNamespace(from_pretrained=lambda path: _PolicyConfig()))
    monkeypatch.setattr(control_utils, "make_policy", lambda **kw: calls.make_policy.append(kw) or _Policy())
    monkeypatch.setattr(
        control_utils, "make_pre_post_processors", lambda **kw: calls.processors.append(kw) or ("pre", "post")
    )
    return calls


def _cfg(rename_map: dict, dataset=None):
    primitive = SimpleNamespace(
        is_adaptive=True, features={}, num_cameras=1, policy=_PolicyConfig(pretrained_path="ckpt"), policy_overwrites={}
    )
    env = SimpleNamespace(primitives={"grab": primitive}, reset_primitive="reset", fps=30, type="test_robot")
    return SimpleNamespace(env=env, dataset=dataset, resume=False, rename_map=rename_map)


def test_the_rename_map_is_a_record_config_field_and_no_longer_a_dataset_field():
    record_fields = {f.name for f in dataclasses.fields(RecordConfig)}
    dataset_fields = {f.name for f in dataclasses.fields(DatasetRecordConfig)}

    assert "rename_map" in record_fields
    assert "rename_map" not in dataset_fields


def test_the_rename_map_defaults_to_empty():
    field = next(f for f in dataclasses.fields(RecordConfig) if f.name == "rename_map")

    assert field.default_factory() == {}


def test_a_policy_is_renamed_without_any_dataset(captured):
    """The case that used to need a dataset config: evaluating a checkpoint whose cameras are named differently."""
    datasets, policies, *_ = control_utils.make_policies_and_datasets(_cfg(RENAME, dataset=None))

    assert datasets == {} and list(policies) == ["grab"]
    assert captured.make_policy[0]["rename_map"] == RENAME
    (processors,) = captured.processors
    assert processors["preprocessor_overrides"]["rename_observations_processor"] == {"rename_map": RENAME}
    assert processors["dataset_stats"] is None


def test_an_empty_rename_map_is_passed_on_as_such(captured):
    control_utils.make_policies_and_datasets(_cfg({}, dataset=None))

    assert captured.make_policy[0]["rename_map"] == {}
    assert captured.processors[0]["preprocessor_overrides"]["rename_observations_processor"] == {"rename_map": {}}
