"""Shared metadata types passed alongside feature dicts.

Models that need extra supervision/context (beyond the float feature dict) should accept an
optional `Meta` instance. Models that don't need it should ignore it.
"""

from __future__ import annotations


class Meta:
  """Marker base class for non-feature metadata passed to models."""

