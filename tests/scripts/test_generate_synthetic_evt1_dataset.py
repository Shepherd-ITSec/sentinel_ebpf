"""Tests for scripts/generate_synthetic_evt1_dataset.py."""

from scripts.generate_synthetic_evt1_dataset import _is_network_row


def test_is_network_row_true_for_network_syscalls():
  assert _is_network_row({"eventName": "connect"}) is True
  assert _is_network_row({"eventName": "socket"}) is True
  assert _is_network_row({"eventName": "sendto"}) is True


def test_is_network_row_false_for_non_network_syscalls():
  assert _is_network_row({"eventName": "openat"}) is False
  assert _is_network_row({"eventName": "close"}) is False
  assert _is_network_row({"eventName": "execve"}) is False
