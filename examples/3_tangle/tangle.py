from canonizator import *
import numpy as np
import matplotlib.pyplot as plt

N = 10

tangles = np.empty(N)
opt_tangles = np.empty(N)

for i in range(N):
    state = create_random_state(i)
    tangles[i] = compute_random_tangle(i)
    print(tangles[i])

    fun, params = canonize(state)
    opt_tangles[i] = canonical_tangle(state, params, post_selection=False)
    print(fun, opt_tangles[i])

fig, ax = plt.subplots()
ax.scatter(tangles, opt_tangles)
ax.plot([0., 1.], [0., 1.], color=black)
plt.show()
