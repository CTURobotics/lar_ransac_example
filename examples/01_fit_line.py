#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-03-15
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#
import numpy as np
import matplotlib.pyplot as plt
from examples.model_line import ModelLine

ref_model = ModelLine(1, 0)

x = np.linspace(-0.5, 0.5, 100)
y = ref_model(x) + np.random.normal(0.0, 0.05, size=x.shape)

model = ModelLine()
model.fit(x, y)

fig, ax = plt.subplots(1, 1, squeeze=True)  # type: plt.Figure, plt.Axes
ax.plot(x, y, "x", color="black", label="data", alpha=0.5)
ax.plot(x, ref_model(x), "-", color="tab:green", label="Reference model", lw=4)

ax.plot(x, model(x), "-", color="tab:blue", label="Fitted model", lw=3)

ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.legend()
fig.savefig("simple_fitting_result.pdf")
plt.show()
