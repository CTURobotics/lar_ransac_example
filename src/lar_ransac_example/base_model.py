#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-03-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
from abc import ABC, abstractmethod
from numpy.typing import ArrayLike


class Model(ABC):
    """Model represent our geometry in a form y = f(x)."""

    @abstractmethod
    def fit(self, x: ArrayLike, y: ArrayLike) -> None:
        """Fit the model internal parameters to the given data."""
        pass

    @abstractmethod
    def copy(self):
        """Copy the model."""
        pass

    @abstractmethod
    def __call__(self, x: ArrayLike, *args, **kwargs):
        """Predict observations for given x."""
        pass
