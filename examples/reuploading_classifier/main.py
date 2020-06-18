import datasets as ds
from qlassifier import single_qubit_classifier

ql = single_qubit_classifier('circle', 1)
result, parameters = ql.minimize(method='l-bfgs-b', options={'disp':True}, compile=True)
ql.set_parameters(parameters)
labels = ql.eval_test_set_fidelity()
for x, y in zip(ql.test_set[0], labels):
    print(x, y)