import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Plooter import *

FixPlot(8, 8)

#2D LinePlot no fit

x = np.linspace(-10, 10, 1000)
y1 = np.sin(x)

fig, ax = plt.subplots()
PlottaLine(ax, x, y1, label="sin(x)")

y2 = np.sin(2.0 * x)
PlottaLine(ax, x, y2, label="sin(2x)")

nY = np.concatenate([y1, y2])
nX = np.concatenate([x, x])

FixScale(ax, nX, nY, pady=1/15, padx=1/8, mirrorx=True, mirrory=True)
FixTicks(ax, 5, 5, 3, 0.2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()

fig.savefig("2dNoFit.png", dpi=500, bbox_inches='tight')

#2D LinePlot with fit

x = np.linspace(-10, 10, 1000)
y = 5.3 * np.sin(x + 2.0) + 1.5

fig, ax = plt.subplots()
popt, pconv, [xb, yb], _, [xf, yf], _ = PlottaLine(ax, x, y, fit=True, funcfit=lambda x, a, b, c, d, e: a * np.sin(b * x + c) * np.exp(-x * e) + d)

nY = np.concatenate([yb, yf])
nX = np.concatenate([xb, xf])

FixScale(ax, nX, nY, pady=1/15, padx=1/8, mirrorx=True, mirrory=True)
FixTicks(ax, 5, 5, 3, 2.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
print("POPT: \n")
print(popt)
print("PCOV: \n")
print(pconv)

fig.savefig("2dWithFit.png", dpi=500, bbox_inches='tight')

#2D Scatter with fit

fig, ax = plt.subplots()

x = 2 * np.random.random(100) - 1.0
y = 2 * np.random.random(100) - 1.0

PlottaScatter(ax, x, y, label=r"$\delta$")

x = 2 * np.random.random(100) - 1.0
y = 2 * np.random.random(100) - 1.0

PlottaScatter(ax, x, y, label=r"$\omega$", color='green')
ax.legend()

fig.savefig("2dScatter.png", dpi=500, bbox_inches='tight')

