"""Tests for scripts/analyze_dataset.py."""

from scripts.analyze_dataset import _downsample_pairs


def test_downsample_pairs_returns_original_when_under_limit():
  xs = [0, 1, 2]
  ys = [0.1, 0.2, 0.3]

  out_xs, out_ys = _downsample_pairs(xs, ys, max_points=5)

  assert out_xs == xs
  assert out_ys == ys


def test_downsample_pairs_keeps_last_point_when_sampling():
  xs = list(range(10))
  ys = [float(i) for i in xs]

  out_xs, out_ys = _downsample_pairs(xs, ys, max_points=4)

  assert len(out_xs) <= 5
  assert out_xs[0] == 0
  assert out_xs[-1] == 9
  assert out_ys[-1] == 9.0
  assert out_xs == sorted(out_xs)


def test_downsample_pairs_zero_limit_disables_sampling():
  xs = list(range(10))
  ys = [float(i) for i in xs]

  out_xs, out_ys = _downsample_pairs(xs, ys, max_points=0)

  assert out_xs == xs
  assert out_ys == ys
