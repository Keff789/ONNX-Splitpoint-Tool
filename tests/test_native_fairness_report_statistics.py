"""The bounded-log A/B fallback retains the collector's small-sample uncertainty."""
import math
import sys

import pytest

from onnx_splitpoint_tool.native_energy_reporting import _repeat_statistics


@pytest.mark.parametrize("without_scipy", [False, True])
@pytest.mark.parametrize("n,confidence,critical", [
    (2, .95, 12.706204736174694),
    (3, .95, 4.302652729749462),
    (4, .95, 3.182446305284263),
    (11, .95, 2.2281388519649385),
    (6, .99, 4.032142983555228),
])
def test_ab_fallback_uses_finite_sample_quantiles(monkeypatch, without_scipy, n, confidence, critical):
    if without_scipy:
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
    result = _repeat_statistics(list(range(n)), confidence)
    assert result["critical_value"] == pytest.approx(critical, rel=1e-8)
    assert result["ci_half_width"] == pytest.approx(critical * result["sample_stddev"] / math.sqrt(n), rel=1e-8)


def test_ab_fallback_counts_only_finite_values_and_no_single_repeat_interval():
    assert _repeat_statistics([])["status"] == "no_valid_repeats"
    assert _repeat_statistics([5.])["ci_low"] is None
    result = _repeat_statistics([9., 10., 11., float("nan"), float("inf")])
    assert result["n"] == 3
    assert result["mean"] == 10.
    assert result["ci_half_width"] == pytest.approx(4.302652729749462 / math.sqrt(3), rel=1e-8)


@pytest.mark.parametrize("confidence", [0, 1, -1, float("nan"), float("inf")])
def test_ab_fallback_does_not_silently_clamp_confidence(confidence):
    with pytest.raises(ValueError):
        _repeat_statistics([1., 2., 3.], confidence)
