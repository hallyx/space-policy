from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

from lerobot_policy_spacepolicy_dit.act_temporal_ensemble import (
    WindowedACTTemporalEnsembler,
    install_windowed_act_temporal_ensemble,
)


@pytest.mark.parametrize("coefficient", [0.0, 0.01, -0.01])
def test_full_window_matches_lerobot_online_implementation(coefficient: float) -> None:
    generator = torch.Generator().manual_seed(1234)
    official = ACTTemporalEnsembler(coefficient, chunk_size=8)
    windowed = WindowedACTTemporalEnsembler(coefficient, chunk_size=8, max_history=8)

    for _ in range(20):
        actions = torch.randn(2, 8, 3, generator=generator)
        expected = official.update(actions)
        actual = windowed.update(actions)
        torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)


def test_one_step_history_is_latest_prediction_action_zero() -> None:
    ensembler = WindowedACTTemporalEnsembler(0.01, chunk_size=4, max_history=1)
    first = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    second = first + 100.0

    torch.testing.assert_close(ensembler.update(first), first[:, 0])
    torch.testing.assert_close(ensembler.update(second), second[:, 0])


def test_coefficient_sign_controls_oldest_vs_newest_weight() -> None:
    old = torch.full((1, 4, 1), 10.0)
    new = torch.zeros((1, 4, 1))
    positive = WindowedACTTemporalEnsembler(1.0, chunk_size=4, max_history=4)
    negative = WindowedACTTemporalEnsembler(-1.0, chunk_size=4, max_history=4)

    positive.update(old)
    negative.update(old)
    assert positive.update(new).item() > 5.0
    assert negative.update(new).item() < 5.0


def test_reset_discards_prediction_history() -> None:
    ensembler = WindowedACTTemporalEnsembler(0.01, chunk_size=4, max_history=3)
    actions = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    ensembler.update(actions)
    ensembler.update(actions + 10.0)
    ensembler.reset()
    torch.testing.assert_close(ensembler.update(actions + 20.0), actions[:, 0] + 20.0)


def test_install_updates_only_runtime_ensemble_contract() -> None:
    class FakePolicy:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                type="act",
                chunk_size=64,
                n_action_steps=8,
                temporal_ensemble_coeff=0.01,
            )
            self.reset_calls = 0

        def reset(self) -> None:
            self.reset_calls += 1
            self.temporal_ensembler.reset()

    policy = FakePolicy()
    installed = install_windowed_act_temporal_ensemble(
        policy,
        temporal_ensemble_coeff=-0.01,
        max_history=8,
    )

    assert policy.temporal_ensembler is installed
    assert policy.config.n_action_steps == 1
    assert policy.config.temporal_ensemble_coeff == pytest.approx(-0.01)
    assert installed.max_history == 8
    assert policy.reset_calls == 1


@pytest.mark.parametrize(
    ("chunk_size", "max_history"),
    [(0, 1), (4, 0), (4, 5)],
)
def test_invalid_history_is_rejected(chunk_size: int, max_history: int) -> None:
    with pytest.raises(ValueError):
        WindowedACTTemporalEnsembler(0.01, chunk_size, max_history)
