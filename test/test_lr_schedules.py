from __future__ import annotations

import math

import pytest

from whittle.lr_schedules import get_wsd_lr

# 10 warmup iterations, then 100 more: with stable_ratio=0.5, the learning rate is
# constant until iteration 60 and decays from 60 to 110.
LR = 1.0
MIN_LR = 0.1
SCHEDULE = {"warmup_iters": 10, "max_iters": 110, "min_lr": MIN_LR, "stable_ratio": 0.5}


def lr_at(it, decay_type="linear"):
    return get_wsd_lr(LR, it, decay_type=decay_type, **SCHEDULE)


def test_warmup_is_linear():
    assert lr_at(0) == 0.0
    assert lr_at(5) == pytest.approx(0.5)


def test_stable_phase_keeps_the_peak():
    assert lr_at(10) == LR
    assert lr_at(59) == LR


@pytest.mark.parametrize(
    "decay_type, expected_midpoint",
    [
        ("linear", MIN_LR + 0.5 * (LR - MIN_LR)),
        ("cosine", MIN_LR + 0.5 * (LR - MIN_LR)),
        ("exponential", LR * math.sqrt(MIN_LR / LR)),
    ],
)
def test_decay(decay_type, expected_midpoint):
    assert lr_at(60, decay_type) == pytest.approx(LR)
    assert lr_at(85, decay_type) == pytest.approx(expected_midpoint)
    assert lr_at(109, decay_type) < lr_at(85, decay_type)


def test_floor_after_max_iters():
    assert lr_at(110) == MIN_LR
    assert lr_at(500) == MIN_LR


def test_unknown_decay_type():
    with pytest.raises(ValueError, match="Unknown decay_type"):
        lr_at(85, decay_type="step")
