"""Shared covariance and matrix-cleaning primitives."""

from .covariance import correlation_to_covariance, covariance_to_correlation, make_psd
from .marchenko_pastur import MarchenkoPasturLaw, marchenko_pastur_law
from .pipeline import estimate_clean_covariance_at_date, estimate_clean_covariance_panel
from .rie import clean_correlation_matrix, eigen_decomposition, supported_cleaning_methods

__all__ = [
    "clean_correlation_matrix",
    "correlation_to_covariance",
    "MarchenkoPasturLaw",
    "covariance_to_correlation",
    "eigen_decomposition",
    "estimate_clean_covariance_at_date",
    "estimate_clean_covariance_panel",
    "make_psd",
    "marchenko_pastur_law",
    "supported_cleaning_methods",
]

