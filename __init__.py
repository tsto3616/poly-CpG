"""
poly-CpG: A self-tuning sliding-window framework for phenotype-linked
regional poly-methylation architecture in sparse wildlife methylomes.

Public API:
    pc.self_tuning_windows_dual()
    pc.classify_windows()
    pc.window_scores()
    pc.r2_and_sign()
    pc.load_example()
"""

from .sliding_window import (
    window_scores,
    r2_and_sign,
    self_tuning_windows_dual,
    classify_windows,
)

from .utils import load_example

__all__ = [
    "window_scores",
    "r2_and_sign",
    "self_tuning_windows_dual",
    "classify_windows",
    "load_example",
]
