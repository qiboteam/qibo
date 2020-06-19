import datasets as ds
import numpy as np
from qlassifier import single_qubit_classifier

ql = single_qubit_classifier('tricrown', 5)
'''result, parameters = ql.minimize(method='l-bfgs-b', options={'disp':True}, compile=True, fidelity=True) # minimizador no va muy allá
print(parameters)'''

params_tricrown = [ 1.71505633,  1.18816839, -0.41820971, -1.17124703, -1.96595341,  1.67279085,
 -2.67943218,  1.75290379,  0.95497698,  1.16126111, -1.48343009, -0.5020429,
  1.13796601, -2.61653729,  0.91361391, -1.64368751,  2.52871188, -1.0190579,
  0.09472308,  1.72974633]
'''params_circle = [ -1.51413155   1.68075594  -3.53392853   1.49516143  -1.35997827
   1.41132915  -9.14847509   3.10432847   0.13614464  -3.34835193
 -12.54425487  -0.05034364   1.36279703  -1.51664904   3.13465347
   1.50204086   1.49518814  -1.65829912  -0.403254     0.94934493]'''
#parameters = np.array([-9.22309909e-07,  1.82987902e+00, -1.65534402e+00,  1.53644454e-01])
#parameters = 20 * np.pi * (.5 - np.random.rand(4))
ql.set_parameters(params_tricrown)
labels = ql.eval_test_set_fidelity()
# ql.paint_results()
ql.paint_world_map()