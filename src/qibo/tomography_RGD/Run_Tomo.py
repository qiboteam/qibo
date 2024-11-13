#
#   prepare the state, do the measurements
#   and directly do the tomography
#
#

import measurements
import methodsMiFGD_core
import methodsRGD_core
import numpy as np
import projectors
import qutip as qu
from BasicTools import Generate_All_labels, Plt_Err_Time
from measurements import Measurement

# from states import GHZState, HadamardState, RandomState
from qibo_states import GHZState, HadamardState, RandomState

from qibo.quantum_info import random_density_matrix
from qibo.models.encodings import ghz_state

def effective_parity(key, label):
    """Calculates the effective number of '1' in the given key

    Args:
        key (str): the measurement outcome in the form of 0 or 1
                (eg)  '101' or '110' for the 3 qubit measurement
        label (str): the label for Pauli measurement
                (eg)  'XIZ', 'YYI', 'XZY' for the 3 qubit measurement
    Returns:
        int: the number of effective '1' in the key
    """
    indices = [i for i, symbol in enumerate(label) if symbol == "I"]
    digit_list = list(key)
    for i in indices:
        digit_list[i] = "0"
    effective_key = "".join(digit_list)

    return effective_key.count("1")


def count_dict_2_PaulCoef(label, count_dict):
    """to convert the shot measurement result for the given Pauli label
        into the coefficient of the label in the Pauli operator basis

    Args:
        label (str): the label for Pauli measurement
            (eg)  'XIYZ', 'XYYI', 'ZXZY' for the 4 qubit measurement
        count_dict (dict): the shot measurement result for the label
            (eg) {'0011': 9, '0100': 7, '1001': 8, '0101': 11, '1101': 3,
                  '0001': 9, '1000': 9, '0000': 12, '1110': 19, '1111': 4,
                  '0111': 1, '1100': 2, '1010': 5, '0110': 1}

    Returns:
        (float): the coefficient in the Pauli operator basis corresponding to the label
    """
    num_shots = sum(count_dict.values())

    freq = {k: (v) / (num_shots) for k, v in count_dict.items()}
    parity_freq = {k: (-1) ** effective_parity(k, label) * v for k, v in freq.items()}
    coef = sum(parity_freq.values())
    # data2 = {label: coef}

    return coef


def Test_measurement(labels, data_dict_list):

    num_labels = len(labels)

    #
    # shot measurement results -->  coefficient for each Pauli operator
    #

    data_dict = {}
    for ii in range(num_labels):
        label = data_dict_list[ii]["label"]
        data_dict[label] = data_dict_list[ii]["count_dict"]

    measurement_list = measurements.MeasurementStore.calc_measurement_list(
        labels, data_dict
    )

    #
    # = measurements.MeasurementStore.calc_measurement_list
    #
    count_dict_list = [data_dict[label] for label in labels]  # in the order of labels
    measurement_list2 = measurements.measurement_list_calc(labels, count_dict_list)
    print(np.array(measurement_list2) - np.array(measurement_list))

    #
    #  how measurements.measurement_list_calc works
    #
    measurement_object_list = [
        Measurement(label, count_dict)
        for (label, count_dict) in zip(*[labels, count_dict_list])
    ]

    parity_flavor = "effective"
    beta = None
    measurement_list3 = [
        measurement_object.get_pauli_correlation_measurement(beta, parity_flavor)[label]
        for (label, measurement_object) in zip(*[labels, measurement_object_list])
    ]
    print(np.array(measurement_list3) - np.array(measurement_list))

    return count_dict_list, measurement_list


if __name__ == "__main__":

    ############################################################
    ### Example of creating and running an experiment
    ############################################################

    # n = 3;    labels = projectors.generate_random_label_list(50, n)
    n = 4
    labels = projectors.generate_random_label_list(120, n)

    # labels = ['YXY', 'IXX', 'ZYI', 'XXX', 'YZZ']
    # labels = ['YZYX', 'ZZIX', 'XXIZ', 'XZIY', 'YXYI', 'ZYYX', 'YXXX', 'IIYY', 'ZIXZ', 'IXXI', 'YZXI', 'ZZYI', 'YZXY', 'XYZI', 'XZXI', 'XZYX', 'YIXI', 'IZYY', 'ZIZX', 'YXXY']
    # labels = ['IIIX', 'IYIY', 'YYXI', 'ZZYY', 'ZYIX', 'XIII', 'XXZI', 'YXZI', 'IZXX', 'YYIZ', 'XXIY', 'XXZY', 'ZZIY', 'YIYX', 'YYZZ', 'YZXZ', 'YZYZ', 'ZXYY', 'IXIZ', 'XZII']
    # labels = Generate_All_labels(n)

    num_labels = len(labels)

    circuit_Choice = 1
    if circuit_Choice == 1:  #  generate from circuit
        Nr = 1

        state   = GHZState(n)
        # state   = HadamardState(n)
        #state = RandomState(n)

        stateGHZ = ghz_state(n)
        target_state_GHZ = stateGHZ.execute().state()

        target_density_matrix = state.get_state_matrix()
        target_state = state.get_state_vector()
        # print(state.get_state_vector())

        #
        # DO the shot measurement
        #

        state.create_circuit()
        data_dict_list = state.execute_measurement_circuits(labels)
        # print(data_dict_list)

        count_dict_list, measurement_list = Test_measurement(labels, data_dict_list)

        #
        # count_dict_list --> measurement_list directly
        #

        measurement_list4 = []

        for label, count_dict in zip(
            *[labels, count_dict_list]
        ):  # count_dict_list in the order of labels

            coef = count_dict_2_PaulCoef(label, count_dict)

            measurement_list4.append(coef)

        print(np.allclose(np.array(measurement_list4), np.array(measurement_list)))
        raise

    elif circuit_Choice == 2:  # directly generate density matrix via qutip
        Nr = 1
        rho = qu.rand_dm_ginibre(2**n, dims=[[2] * n, [2] * n], rank=Nr)
        target_density_matrix = rho.full()

    elif circuit_Choice == 3:  # directly generate density matrix via qibo
        Nr = 3
        target_density_matrix = random_density_matrix(2**n, Nr)

    # ---------------------------------------------------------------- #
    #  construct Pauli matrix Projectors                               #
    # ---------------------------------------------------------------- #

    projector_store_path = "./testData/qiskit"
    # projector_store_path = './testData/qibo'
    projector_store = projectors.ProjectorStore(labels)

    ## store method (1)
    # projector_store.populate(projector_store_path)
    # projector_dict = projectors.ProjectorStore.load(projector_store_path, labels)

    ## store method (2)
    num_cpus, saveP_bulk, Partition_Pj = projector_store.mpPool_map(
        projector_store_path
    )
    projector_dict = projectors.ProjectorStore.load_PoolMap(
        projector_store_path, labels
    )

    # ---------------------------------------------------------------- #
    #  calculate exact coefficient for each Pauli operator             #
    # ---------------------------------------------------------------- #

    projector_list = [projector_dict[label] for label in labels]
    yProj_Exact = methodsRGD_core.Amea(
        projector_list, target_density_matrix, num_labels, 1
    )  # argv[-1] = coef

    if circuit_Choice == 1:  #  generated from circuit
        #
        #   comparison: manual check the result of shot measurements
        #
        ml = np.array(measurement_list, dtype=float)
        ind = np.where(np.abs(yProj_Exact) > 0.5)

        print(yProj_Exact[ind])
        print(ml[ind])
    elif circuit_Choice == 2 or circuit_Choice == 3:  # directly generate density matrix
        measurement_list = yProj_Exact
        target_state = None

    #
    #   system parameters
    #
    params_dict = {
        "Nr": Nr,
        "target_DM": target_density_matrix,
        "labels": labels,
        "measurement_list": measurement_list,
        "projector_list": projector_list,
        "num_iterations": 150,
        "convergence_check_period": 1,
    }
    # params_dict['target_state'] = target_state

    # ----------------------------------------------------------------- #
    #   do the tomography optimization                                  #
    # ----------------------------------------------------------------- #

    #
    #   MiFGD numerical parameters
    #

    # Call_MiFGD = 1      #  = 1: call the MiFGD optimization to calculate
    # muList = [4.5e-5]

    InitX_MiFGD = 1  # 0: random start,  1: MiFGD specified init
    mu = 4.5e-5
    eta = 0.01
    Option = 2

    # Num_mu = 1      #  number of mu for running MiFGD
    # pm_MiFGD = [Call_MiFGD, InitX_MiFGD, muList, Num_mu]
    # Call_MiFGD, InitX_MiFGD, muList, Num_mu = pm_MiFGD

    params_MiFGD = {"mu": mu, "eta": eta, "Option": Option}
    # params_dict = {**params_dict, **params_MiFGD}

    # Rpm_MiFGD = [InitX_MiFGD, mu, eta, Option]
    # Frec_MiFGD, wc, RunTime = Run_MiFGD(params_dict, Rpm_MiFGD)

    worker2 = methodsMiFGD_core.BasicWorker(params_dict, params_MiFGD)
    worker2.compute(InitX_MiFGD)

    #
    #   RGD numerical parameters
    #

    print("\n +++++++++++++++++     do the RGD tomography     +++++++++++++++++\n")

    # Call_RGD = 1        #  = 1: call the RGD   optimization to calculate

    Md_tr = 0  #   Method if including trace = 1
    Md_alp = 0  #   method for scaling alpha
    Md_sig = 0  #   method for scaling singular value
    Ch_svd = (
        -1
    )  #   choice for initial SVD  (0: LA.svd, 1: svds;  2: power_Largest_EigV, -1: rSVD)
    InitX_RGD = 1
    #   method of choosing initial X0

    # Rpm    = [InitX_RGD, Md_tr, Md_alp, Md_sig, Ch_svd]
    # pm_RGD = [Call_RGD, Rpm]
    # Call_RGD, Rpm   = pm_RGD

    # InitX_RGD, Md_tr, Md_alp, Md_sig, Ch_svd = Rpm
    # InitX_RGD = 1
    # Rpm = InitX_RGD, Md_tr, Md_alp, Md_sig, Ch_svd

    # exec(open('RGD_optRun.py').read())
    # Frec_RGD, wc, RunTime = Run_RGD(params_dict, Rpm)

    worker = methodsRGD_core.BasicWorkerRGD(params_dict)
    # worker.computeRGD(InitX_RGD, Ch_svd, Md_tr, Md_alp, Md_sig)
    worker.computeRGD(InitX_RGD, Ch_svd)

    Plt_Err_Time(worker)
