from tangle_aux import *
import time
seed = 'GHZ'
for t in range(6):
    folder = 'results_simulation/seed_{}/error_{}/'.format(seed, t)
    f = np.loadtxt(folder + 'opti_info.txt')[0]
    if f > t * 0.02 + 0.001:
        print('seed', seed)
        print('error', t)
        write_optimizer_sampling(seed, t)
        write_tangles(seed, t, norm=False)
        write_tangles(seed, t, norm=True)
measured_tangles(1000, error=0, tol=0.0)
#relative_errors(1000, 5, tol=0.02)
compute_chi(1000, max_error=5, tol=0.02)

