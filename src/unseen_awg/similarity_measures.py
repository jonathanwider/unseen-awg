"""Similarity measures for weather generator analog selection.

This module provides similarity measures used to compare reference points
with candidate points in the weather generator's analog selection process.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def mse_similarity(
    ref_point_data: NDArray[np.floating[Any]],
    cand_points_data: NDArray[np.floating[Any]],
    reduction_axes: tuple[int, ...] = (-3, -2, -1),
) -> NDArray[np.floating[Any]]:
    """Calculate similarity based on negative mean squared error.

    Computes the similarity between a reference point and candidate points
    using the negative mean squared error. Lower MSE values result in higher
    (less negative) similarity scores.

    Parameters
    ----------
    ref_point_data : NDArray[np.floating[Any]]
        Reference point data array.
    cand_points_data : NDArray[np.floating[Any]]
        Candidate points data array with the same shape as ref_point_data
        or broadcastable to it.
    reduction_axes : tuple[int, ...], optional
        Axes along which to compute the mean, by default (-3, -2, -1).
        These should correspond to (latitude, longitude, lag).

    Returns
    -------
    NDArray[np.floating[Any]]
        Similarity scores as negative MSE values. Higher values indicate
        greater similarity.

    Notes
    -----
    The similarity is computed as:
    similarity = -mean((cand_points_data - ref_point_data)^2)
    """
    squared_diff = (cand_points_data - ref_point_data) ** 2
    mse = np.mean(squared_diff, axis=reduction_axes)
    similarity = -mse
    return similarity
