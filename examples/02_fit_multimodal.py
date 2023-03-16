#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-03-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np
import matplotlib.pyplot as plt

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

model = ModelLine()
model.fit(x, y)

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(x, y, "x", color="black", label="data", alpha=0.5)
ax.plot(x1, ref_model1(x1), "-", color="tab:green", label="Reference model 1", lw=4)
ax.plot(x2, ref_model2(x2), "-", color="tab:green", label="Reference model 2", lw=4)

ax.plot(x1, model(x1), "-", color="tab:blue", label="Fitted model", lw=3)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.legend()
fig.savefig("multimodal_fitting_result.pdf")
plt.show()
