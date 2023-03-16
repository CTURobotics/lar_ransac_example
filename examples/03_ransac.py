#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-03-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

import numpy as np
import matplotlib.pyplot as plt
from lar_ransac_example import ransac

from model_line import ModelLine


ref_model1 = ModelLine(1, 0)
ref_model2 = ModelLine(-1, 0.5)

x = np.linspace(-0.5, 0.5, 100)
x1 = np.linspace(-0.5, 0.5, 100)
y1 = ref_model1(x) + np.random.normal(0.0, 0.05, size=x1.shape)
x2 = np.linspace(-0.5, 0.5, 100)
y2 = ref_model2(x) + np.random.normal(0.0, 0.05, size=x2.shape)

x = np.concatenate([x1, x2])
y = np.concatenate([y1, y2])

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(x, y, "x", color="black", label="data", alpha=0.5)
ax.plot(x1, ref_model1(x1), "-", color="tab:green", label="Reference model 1", lw=4)
ax.plot(x2, ref_model2(x2), "-", color="tab:green", label="Reference model 2", lw=4)

model1, model1_indicies = ransac(ModelLine(), x, y)

ax.plot(x1, model1(x1), "-", color="tab:blue", label="Ransac1", lw=3)
ax.plot(
    x[model1_indicies],
    y[model1_indicies],
    "o",
    color="tab:blue",
    label="Ransac1 Inliers",
    alpha=0.5,
)
ax.legend()
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
fig.savefig("ransac_fitting_result1.pdf")

x, y = x[np.bitwise_not(model1_indicies)], y[np.bitwise_not(model1_indicies)]
model2, model2_indicies = ransac(ModelLine(), x, y)

ax.plot(x1, model2(x1), "-", color="tab:red", label="Ransac2", lw=3)
ax.plot(
    x[model2_indicies],
    y[model2_indicies],
    "o",
    color="tab:red",
    label="Ransac2 Inliers",
    alpha=0.5,
)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.legend()
fig.savefig("ransac_fitting_result2.pdf")

plt.show()
