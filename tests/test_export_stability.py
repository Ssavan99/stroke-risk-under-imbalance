"""The exported web payload has to be byte-identical on every machine.

The Pages workflow guards against a stale ``web/`` directory by rebuilding the
model and comparing it to what is committed. The first version of that guard
compared bytes, which failed the moment CI's BLAS disagreed with a laptop's in
the sixteenth significant digit -- a difference of about 2e-16 reported as
model drift. These tests pin both halves of the fix: floats are quantised on
export so the artefact is reproducible, and the staleness check compares
numerically so a rounding tie cannot fail a build on its own.
"""

from __future__ import annotations

import json
import math

import pytest

from stroke.export_web import EXPORT_SIG_DIGITS, _compare, _quantize, build_payload


@pytest.fixture(scope="module")
def payload():
    return _quantize(build_payload())


def test_quantised_floats_survive_a_json_round_trip(payload):
    """Serialising and reparsing must not change a single exported value."""
    reparsed = json.loads(json.dumps(payload, indent=2))
    diffs: list[str] = []
    _compare(payload, reparsed, "payload", rtol=0.0, atol=0.0, out=diffs)
    assert diffs == []


def test_quantisation_absorbs_last_place_noise():
    """One ULP of BLAS disagreement must quantise away, not survive into the file."""
    value = 0.002264495627693729
    nudged = math.nextafter(value, math.inf)
    assert value != nudged
    assert _quantize(value) == _quantize(nudged)


def test_quantisation_keeps_far_more_precision_than_the_page_shows():
    """Rounding must not perturb a probability anywhere near a visible digit."""
    value = 0.08412718632429177
    assert abs(_quantize(value) - value) < 10 ** -(EXPORT_SIG_DIGITS - 2)


def test_comparison_tolerates_ulp_noise_but_catches_real_drift():
    """The guard's whole point: insensitive to noise, sensitive to change."""
    base = {"coefficients": [1.62990942507, -0.5], "intercept": 0.25}

    noisy = {
        "coefficients": [math.nextafter(v, math.inf) for v in base["coefficients"]],
        "intercept": math.nextafter(base["intercept"], math.inf),
    }
    noise_diffs: list[str] = []
    _compare(base, noisy, "m", rtol=1e-6, atol=1e-12, out=noise_diffs)
    assert noise_diffs == []

    drifted = {**base, "coefficients": [base["coefficients"][0] * 1.01, -0.5]}
    drift_diffs: list[str] = []
    _compare(base, drifted, "m", rtol=1e-6, atol=1e-12, out=drift_diffs)
    assert len(drift_diffs) == 1
    assert "coefficients[0]" in drift_diffs[0]


def test_comparison_reports_structural_changes():
    """A renamed or dropped key is real drift, not numerical noise."""
    diffs: list[str] = []
    _compare({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0}, "m", 1e-6, 1e-12, diffs)
    assert len(diffs) == 2

    length: list[str] = []
    _compare({"a": [1.0, 2.0]}, {"a": [1.0]}, "m", 1e-6, 1e-12, length)
    assert length and "length" in length[0]
