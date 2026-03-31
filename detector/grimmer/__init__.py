"""Grimmer / LID-DS-aligned syscall sequence pipeline (thread-aware n-grams, Word2Vec, MLP).

Ported and adapted from the LID-DS reference implementation under ``third_party/LID-DS``
(GPL-3.0). See module docstrings for file-level citations.
"""

from detector.grimmer.pipeline import GrimmerPipeline

__all__ = ["GrimmerPipeline"]
