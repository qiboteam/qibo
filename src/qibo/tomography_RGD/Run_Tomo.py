#
#   prepare the state, do the measurements
#   and directly do the tomography
#
#

import numpy as np
import qutip as qu

from BasicTools import Plt_Err_Time

#from states import GHZState, HadamardState, RandomState
from qibo_states import GHZState, HadamardState, RandomState
from qibo import quantum_info


import methodsRGD_core
import methodsMiFGD_core

import measurements
import projectors


from BasicTools import Generate_All_labels


if __name__ == '__main__':

    ############################################################
    ### Example of creating and running an experiment
    ############################################################

    #n = 3;    labels = projectors.generate_random_label_list(50, n)
    n = 4;    labels = projectors.generate_random_label_list(120, n)

    #labels = ['YXY', 'IXX', 'ZYI', 'XXX', 'YZZ']
    #labels = ['YZYX', 'ZZIX', 'XXIZ', 'XZIY', 'YXYI', 'ZYYX', 'YXXX', 'IIYY', 'ZIXZ', 'IXXI', 'YZXI', 'ZZYI', 'YZXY', 'XYZI', 'XZXI', 'XZYX', 'YIXI', 'IZYY', 'ZIZX', 'YXXY']
    #labels = ['IIIX', 'IYIY', 'YYXI', 'ZZYY', 'ZYIX', 'XIII', 'XXZI', 'YXZI', 'IZXX', 'YYIZ', 'XXIY', 'XXZY', 'ZZIY', 'YIYX', 'YYZZ', 'YZXZ', 'YZYZ', 'ZXYY', 'IXIZ', 'XZII']
    #labels = Generate_All_labels(n)

    num_labels  = len(labels)

    circuit_Choice = 1
    if circuit_Choice == 1:             #  generate from circuit
        Nr = 1 

        #state   = GHZState(n)
        #state   = HadamardState(n)
        state   = RandomState(n)

        target_density_matrix = state.get_state_matrix()
        target_state          = state.get_state_vector()            
        #print(state.get_state_vector())

        #
        # DO the shot measurement
        #

        state.create_circuit()
        data_dict_list = state.execute_measurement_circuits(labels)
        #print(data_dict_list)

        #
        # shot measurement results -->  coefficient for each Pauli operator
        #

        data_dict = {}
        for ii in range(num_labels):
            label = data_dict_list[ii]['label']
            data_dict[label] = data_dict_list[ii]['count_dict'] 
                
        measurement_list = measurements.MeasurementStore.calc_measurement_list(labels, data_dict)


    elif circuit_Choice == 2:            # directly generate density matrix via qutip
        Nr = 1
        rho = qu.rand_dm_ginibre(2**n, dims=[[2]*n , [2]*n], rank=Nr)
        target_density_matrix = rho.full()

    elif circuit_Choice == 3:            # directly generate density matrix via qibo
        Nr = 3
        target_density_matrix = quantum_info.random_density_matrix(2**n, Nr)

    # ---------------------------------------------------------------- #
    #  construct Pauli matrix Projectors                               #
    # ---------------------------------------------------------------- #

    projector_store_path = './testData/qiskit'
    #projector_store_path = './testData/qibo'
    projector_store = projectors.ProjectorStore(labels)

    ## store method (1)
    #projector_store.populate(projector_store_path)
    #projector_dict = projectors.ProjectorStore.load(projector_store_path, labels)

    ## store method (2)
    num_cpus, saveP_bulk, Partition_Pj = projector_store.mpPool_map(projector_store_path)
    projector_dict = projectors.ProjectorStore.load_PoolMap(projector_store_path, labels)

    # ---------------------------------------------------------------- #
    #  calculate exact coefficient for each Pauli operator             #
    # ---------------------------------------------------------------- #

    projector_list = [projector_dict[label] for label in labels]
    yProj_Exact = methodsRGD_core.Amea(projector_list, target_density_matrix, num_labels, 1) # argv[-1] = coef

    if circuit_Choice == 1:         #  generated from circuit
        #
        #   comparison: manual check the result of shot measurements
        #
        ml = np.array(measurement_list, dtype=float)
        ind = np.where(np.abs(yProj_Exact)>0.5)

        print(yProj_Exact[ind])
        print(ml[ind])
    elif circuit_Choice==2 or circuit_Choice==3:            # directly generate density matrix
        measurement_list = yProj_Exact
        target_state     = None

    #
    #   system parameters
    #
    params_dict = { 'Nr': Nr,
                    'target_DM': target_density_matrix,
                    'labels': labels,
                    'measurement_list': measurement_list,
                    'projector_list': projector_list,
                    'num_iterations': 150,
                    'convergence_check_period': 1 
    }
    #params_dict['target_state'] = target_state         



    # ----------------------------------------------------------------- #
    #   do the tomography optimization                                  #
    # ----------------------------------------------------------------- #

    #
    #   MiFGD numerical parameters
    #
    
    #Call_MiFGD = 1      #  = 1: call the MiFGD optimization to calculate
    #muList = [4.5e-5]


    InitX_MiFGD = 1       # 0: random start,  1: MiFGD specified init    
    mu  = 4.5e-5
    eta = 0.01
    Option = 2

    #Num_mu = 1      #  number of mu for running MiFGD
    #pm_MiFGD = [Call_MiFGD, InitX_MiFGD, muList, Num_mu]
    #Call_MiFGD, InitX_MiFGD, muList, Num_mu = pm_MiFGD

    params_MiFGD = {'mu': mu,
                    'eta': eta,
                    'Option': Option
                    }
    #params_dict = {**params_dict, **params_MiFGD}


    #Rpm_MiFGD = [InitX_MiFGD, mu, eta, Option]
    #Frec_MiFGD, wc, RunTime = Run_MiFGD(params_dict, Rpm_MiFGD)

    worker2 = methodsMiFGD_core.BasicWorker(params_dict, params_MiFGD)
    worker2.compute(InitX_MiFGD)



    #
    #   RGD numerical parameters
    #  

    print('\n +++++++++++++++++     do the RGD tomography     +++++++++++++++++\n')

    #Call_RGD = 1        #  = 1: call the RGD   optimization to calculate

    Md_tr =  0                  #   Method if including trace = 1
    Md_alp = 0                  #   method for scaling alpha
    Md_sig = 0                  #   method for scaling singular value
    Ch_svd = -1                 #   choice for initial SVD  (0: LA.svd, 1: svds;  2: power_Largest_EigV, -1: rSVD)
    InitX_RGD = 1;              #   method of choosing initial X0

    #Rpm    = [InitX_RGD, Md_tr, Md_alp, Md_sig, Ch_svd]
    #pm_RGD = [Call_RGD, Rpm]
    #Call_RGD, Rpm   = pm_RGD

    #InitX_RGD, Md_tr, Md_alp, Md_sig, Ch_svd = Rpm
    #InitX_RGD = 1
    #Rpm = InitX_RGD, Md_tr, Md_alp, Md_sig, Ch_svd

    #exec(open('RGD_optRun.py').read())
    #Frec_RGD, wc, RunTime = Run_RGD(params_dict, Rpm)

    worker = methodsRGD_core.BasicWorkerRGD(params_dict)
    #worker.computeRGD(InitX_RGD, Ch_svd, Md_tr, Md_alp, Md_sig)
    worker.computeRGD(InitX_RGD, Ch_svd)


    Plt_Err_Time(worker)

