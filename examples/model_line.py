#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-03-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from lar_ransac_example import Model


class ModelLine(Model):
    def __init__(self, a: float = 0.1, b: float = 0.1) -> None:
        """Implements a line model in a form y = a*x + b"""
        super().__init__()
        self.a = a
        self.b = b

    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        x = np.asarray(x)
        y = np.asarray(y)
        assert len(x.shape) == 1
        assert len(y.shape) == 1
        assert x.shape[0] > 2
        assert x.shape[0] == y.shape[0]

        popt, _ = curve_fit(self, x, y)
        self.a, self.b = popt

    def copy(self):
        return ModelLine(self.a, self.b)

    def __call__(self, x: ArrayLike, new_a: float = None, new_b: float = None):
        a = self.a if new_a is None else new_a
        b = self.b if new_b is None else new_b
        return x * a + b
