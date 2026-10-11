"""Reviewer-facing baseline implementations used by the paper experiments."""

from .classical import NominalArray, l_shaped_esprit, music_2d, sample_covariance, source_count_mdl
from .neural_set import NeuralSetBaseline, load_setnet_checkpoint
from .sbl import CovarianceSBL

__all__ = [
    "CovarianceSBL",
    "NeuralSetBaseline",
    "NominalArray",
    "l_shaped_esprit",
    "load_setnet_checkpoint",
    "music_2d",
    "sample_covariance",
    "source_count_mdl",
]
