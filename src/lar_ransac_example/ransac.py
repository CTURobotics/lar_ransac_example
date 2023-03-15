#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-03-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

from .base_model import Model
import numpy as np
from numpy.typing import ArrayLike


def ransac(
    model: Model,
    x: ArrayLike,
    y: ArrayLike,
    n_iterations=100,
    n_fitting_size=10,
    inlier_threshold=0.05,
) -> tuple[Model, np.ndarray]:
    """
    Perform ransac fitting on a given model.
    Args:
        model: a model that inherits from base_model.Model class
        x: data points
        y: observation for data points
        n_iterations: number of iterations of ransac
        n_fitting_size: size of a subset of data needed to fit the model
        inlier_threshold: distance for which data is considered to be inlier, measured
        as L2 norm.

    Returns:
        the best model and indicies of inliers
    """
    x, y = np.asarray(x), np.asarray(y)
    n = x.shape[0]
    assert n > n_fitting_size

    best_model = model.copy()
    best_model_number_of_inliers = -1
    best_model_ind_inliers = None
    for _ in range(n_iterations):
        # Choose a subset of points
        ind = np.random.choice(range(n), size=n_fitting_size, replace=False)
        x_fit, y_fit = x[ind], y[ind]

        # Fit a model to that subset
        model.fit(x_fit, y_fit)

        # get number of inliers
        d = np.abs(y - model(x))
        ind_inliers = d < inlier_threshold
        number_of_inliers = np.sum(ind_inliers)

        # store best if needed
        if number_of_inliers > best_model_number_of_inliers:
            best_model_number_of_inliers = number_of_inliers
            best_model = model.copy()
            best_model_ind_inliers = ind_inliers

    return best_model, best_model_ind_inliers
