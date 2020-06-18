import datasets as ds
from qlassifier import single_qubit_classifier

ql = single_qubit_classifier('circle', 1)
result, parameters = ql.minimize(method='l-bfgs-b', options={'disp':True}, compile=True)