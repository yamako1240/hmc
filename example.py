import hmc
import numpy as np
import matplotlib.pyplot as plt


def normal(x):
    return -np.sum(x**2), -2 * x


res = hmc.mc(
    normal,
    np.array([0.5, 0.0]),
    method="nutsda",
    options={
        "delta": 0.5,
        "M": 4000,
        "Madapt": 400,
        "e": 0.2,
        "alltrees": False,
        "L": 5,
         },
)

hist, bin_edges = np.histogram(np.sqrt(np.sum(res[100:] ** 2, axis=1)), bins=50)
# hist, bin_edges = np.histogram(res[0][1000:,0], bins=50)
hist = np.cumsum(hist)
hist = hist / hist[-1]

fig, ax = plt.subplots()
ax.plot(bin_edges[1:-1], np.diff(hist))
ax.plot(bin_edges[1:-1], np.diff(1 - np.exp(-bin_edges[1:] ** 2)), label='analytic solution')
ax.legend()
fig.savefig("sample.pdf")

