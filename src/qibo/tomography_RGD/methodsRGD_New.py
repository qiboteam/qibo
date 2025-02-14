import pickle

# import scipy.sparse as sparse
import time
from functools import reduce

import numpy as np
import projectors
import scipy.sparse as sparse
from numpy import linalg as LA

# from mpi4py import MPI
from qiskit.quantum_info import state_fidelity
from scipy.sparse.linalg import svds

# import measurements


# --------------------------------- #
# 	the origianl RGD package		#
# --------------------------------- #


############################################################
## Basic sequential RGD
## XXX WIP
############################################################


class LoopRGD:
    def __init__(self):
        self.method = "RGD"

    def initialize_RndSt(self):
        """
        Random initialization of the state density matrix.
        """

        print("\n ******  initial choice same as the MiFGD  ***** \n")
        print("self.seed = {}".format(self.seed))
        self.InitChoice = "random initial (same as MiFGD)"

        seed = self.seed

        stateIni = []
        for xx in range(self.Nr):
            real_state = np.random.RandomState(seed).randn(self.num_elements)

            # seed += 1
            imag_state = np.random.RandomState(seed).randn(self.num_elements)

            seed += 1

            # real_state = np.random.RandomState().randn(self.num_elements)
            # imag_state = np.random.RandomState().randn(self.num_elements)

            state = real_state + 1.0j * imag_state

            # state      = 1.0 / np.sqrt(self.num_tomography_labels) * state
            state_norm = np.linalg.norm(state)
            state = state / state_norm

            stateIni.append(state)

        stateIni = np.array(stateIni).T

        self.state = stateIni
        self.uk = stateIni
        self.vk = stateIni
        self.ukh = stateIni.T.conj()
        self.vkh = np.transpose(stateIni.conj())

        self.s0_choice = 0
        if self.s0_choice == 0:
            # ss = np.random.random(self.Nr)
            # ss = ss / np.sum(ss)
            ss = np.ones(self.Nr)

        elif self.s0_choice == 1:
            ss = np.ones(self.Nr) / self.Nr

        self.sDiag = np.diag(ss)

        X0 = stateIni @ self.sDiag @ self.vkh

        self.u0 = stateIni
        self.v0 = stateIni
        self.s0 = np.diag(ss)
        self.X0 = X0

        self.Xk = X0

        self.check_Hermitian_x_y(self.uk, self.vk)

    def initialize_yAd(self):
        """initialize the initial density matrix for the algorithm"""

        self.InitChoice = "paper"

        # ------------------------------------------------------------ #

        if self.Ch_svd >= 0:  #   need to compute X0

            print("  start running self.A_dagger")
            tt1 = time.time()
            # y = self.yIni 	   ##  this y = coef * measurement_list -> order follows self.label_list

            if self.Md_tr == 0:  #  the normal def of A, i.e. sampling all Pauli
                Xk = self.A_dagger(self.yIni)  #  1st quickest

            tt2 = time.time()

            print(" self.A_dagger done  -->  time = {}".format(tt2 - tt1))
            print(" Initial SVD -->  choice = {}".format(self.Ch_svd))

            uk, sDiag, vkh = Hr_Hermitian(
                Xk, self.Nr, self.Ch_svd
            )  #  Ch_svd =0: full SVD

        elif self.Ch_svd < 0:  #   don't need to compute X0

            print(
                " *********   using randomized-SVD  to construct  X0 = uk @ sDiag @ vkh  **************************"
            )

            tt2a = time.time()
            uk, sDiag, vkh = self.rSVD(k=self.Nr, s=2, p=15)
            tt2b = time.time()
            print("   self.rSVD           -->   time = {}".format(tt2b - tt2a))

            # Compare_2_SVD(u1, sDiag1, v1h, uk, sDiag, vkh)

        self.check_Hermitian_x_y(uk, vkh.T.conj())

        if self.EigV_positive == 1:  #   U == V
            vk = uk
            vkh = uk.T.conj()

        else:  #   U != V
            vk = vkh.T.conj()

        ukh = uk.T.conj()
        Xk = uk @ sDiag @ vkh  # only keep rank r singular vectors

        self.u0 = uk
        self.v0 = vk
        self.s0 = sDiag
        self.X0 = Xk

        self.Xk = Xk
        self.uk = uk
        self.vk = vk

        self.ukh = ukh
        self.vkh = vkh
        self.sDiag = sDiag


    def A_dagger_ym_vec(self, zm, vec):
        """to obtain A^+(zm) @ vec

        Args:
                zm (ndarray): the input vector for the A^+ operator
                vec (ndarray): a vector which is supposed to be the right singular vector

        Returns:
                ndarray: Xu = the A^+(y) @ vec
        """

        Xu = np.zeros((self.num_elements, vec.shape[1]), dtype=complex)
        for yP, proj in zip(zm, self.projector_list):
            Xu += yP * proj.dot(vec)
        # Xu *= self.coef

        return Xu

    def rSVD(self, k, s=3, p=12, matrixM_Hermitian=1):
        """randomized SVD

                ref:  https://arxiv.org/pdf/1810.06860.pdf    -->  A ~ QQ* A          (B = Q* A  )
                        with modification:  if Remain_Hermitian   -->  A ~ QQ* A  QQ*     (B = Q* A Q)

        Args:
                k (int): = Nr        # rank parameter
                s (int, optional): specify oversampling. Defaults to 3.
                p (int, optional): specify power parameter. Defaults to 12.
                matrixM_Hermitian (int, optional): specify the matrix is Hermitian or not. Defaults to 1.

        Returns:
                ndarray: (u) left singular vectors
                ndarray: np.diag(s) the diagonal matrix with diagonal elements being s
                ndarray: (vh) complex conjugate of the right singular vectors
        """

        qDs = min(k + s, self.num_elements)  # k + s small dimension for Q

        Omega = np.random.RandomState().randn(self.num_elements, qDs)

        Mw = self.A_dagger_ym_vec(self.yIni, Omega)

        Q, R = LA.qr(Mw)

        for ii in range(p):
            tt1 = time.time()

            ATQ = self.A_dagger_ym_vec(self.yIni, Q)

            G, R = LA.qr(ATQ)
            tt2 = time.time()

            AG = self.A_dagger_ym_vec(self.yIni, G)

            Q, R = LA.qr(AG)

            tt3 = time.time()
            print(
                "  ***   {}-th (A A*) done:   Time --> ATQ: {},  AG: {}   ***".format(
                    ii, tt2 - tt1, tt3 - tt2
                )
            )

        if matrixM_Hermitian == 0:
            # B = Q.T.conj() @ M
            #B = self.A_dagger_vec(Q).T.conj()
            B = self.A_dagger_ym_vec(self.yIni, Q).T.conj() * self.coef

            uB, s, vh = LA.svd(B)
            u = Q @ uB[:, :k]  #  k = Nr
            s = s[:k]  #  k = Nr
            vh = vh[:k, :]  #  k = Nr

        elif matrixM_Hermitian == 1:
            # B = Q.T.conj() @ M @  Q
            #B = Q.T.conj() @ self.A_dagger_vec(Q)
            B = Q.T.conj() @ self.A_dagger_ym_vec(self.yIni, Q) * self.coef

            uB, s, vh = LA.svd(B, hermitian=True)
            u = Q @ uB[:, :k]  #  k = Nr
            s = s[:k]  #  k = Nr

            vh = vh[:k, :] @ Q.T.conj()  #  k = Nr

        return u, np.diag(s), vh

    def check_Hermitian_x_y(self, x, y, Hermitian_criteria=1e-13):
        """to check wheter x and y are the same,
                where x and y are supposed to be the left and right singular vectors
                if x = y, then the original SVDed matrix is Hermitian

        Args:
                x (ndarray): the 1st vector to compare, supposed to be the left singular vector
                y (ndarray): the 2nd vector to compare, supposed to be the right singular vector
                Hermitian_criteria (float, optional): the numerical error minimum criteria. Defaults to 1e-13.
        """

        # if LA.norm(x - y) < Hermitian_criteria:
        if np.allclose(x, y):

            self.Remain_Hermitian = 1  # 	True  for remaining Hermitian
            self.EigV_positive = 1

        # elif LA.norm(Col_flip_sign(x) - Col_flip_sign(y)) < Hermitian_criteria:
        elif np.allclose(Col_flip_sign(x), Col_flip_sign(y)):
            self.Remain_Hermitian = 1
            self.EigV_positive = 0

        else:
            self.EigV_positive = None
            self.Remain_Hermitian = 0  # 	False for remaining Hermitian
            self.st_Non_Hermitian.append(
                self.iteration
            )  #   start num for non-Hermitian

        self.EigV_pm_List.append(self.EigV_positive)
        self.Hermitian_List.append(self.Remain_Hermitian)

        #print(
        #    "            Hermitian ?    Remain_Hermiain = {},    EigV_postive = {}".format(
        #        self.Remain_Hermitian, self.EigV_positive
        #    )
        #)

    def Get_measured_y(self):
        """to get the coefficients of all sampled Pauli operators for the A^+ operator

        """

        coef = np.sqrt(self.num_elements / self.num_labels)
        self.yIni = np.array(self.measurement_list) * coef
        self.coef = coef

    def computeRGD(self, InitX, Ch_svd=0, Md_tr=0, Ld=0):
        """Basic workflow of gradient descent iteration.

        1. initializes state dnesity matrix.
        2. mapkes a step (defined differently for each "Worker" class below) until
           convergence criterion is satisfied.

        Args:
                InitX (int): choice of the initial matrix
                Ch_svd (int, optional): _description_. Defaults to 0.
                Md_tr (int, optional): Method if including trace = 1. Defaults to 0.
                                Md_tr = 0,  the usual RGD algorithm, i.e. sampling all S
                                          = 1,  to enforce coef for Identiy, i.e. not sampling Identity
                Ld (str, optional): specification how to load the data from file. Defaults to 0.
        """
        # self.Id = np.eye(self.num_elements, dtype=complex)  ##  matrix dimension d = num_elements = 2**Nk

        self.InitX = InitX
        self.Md_tr = Md_tr  #  method of including identity operator or not

        self.Ch_svd = Ch_svd  #  choice for initial SVD

        self.EigV_pm_List = (
            []
        )  #  Eig-value is positive or not  (=1: positive,  = 0, non-positive, = None, no EigV)
        self.Hermitian_List = []
        self.st_Non_Hermitian = []
        self.step_Time = {}

        tt1 = time.time()

        self.iteration = -1

        self.Get_measured_y()
        print("  self.coef = {}".format(self.coef))
        print("  InitX     = {}".format(InitX))

        if InitX == 0:
            self.initialize_RndSt()  #  initial by random
        elif InitX == 1:
            self.initialize_yAd()  #  initial by A^\dagger(y)

        tt2 = time.time()
        self.step_Time[-1] = tt2 - tt1

        print(
            " --------  X0  --> (-1)-th iteration  [ InitX = {}, SVD choice = {} ] \n".format(
                InitX, Ch_svd
            )
        )
        print("      X0 step (RGD)        -->  time = {}\n".format(tt2 - tt1))

        X0_target_Err = LA.norm(self.Xk - self.target_DM, "fro")
        self.Target_Err_Xk.append(X0_target_Err)  #  for X0
        self.InitErr = X0_target_Err

        print(
            "      X0                   -->  Tr(X0)     = {}\n".format(
                np.trace(self.X0)
            )
        )
        print("      X0                   -->  Target Err = {}\n".format(X0_target_Err))
        print("    RGD max num_iterations = {}\n".format(self.num_iterations))

        self.uGv_list = []
        self.Rec_Alpha = []
        self.Alpha_sc = []
        self.sDiag_list = []

        for self.iteration in range(self.num_iterations):
            if not self.converged:
                self.stepRGD()
            else:
                break
        if self.convergence_iteration == 0:
            self.convergence_iteration = self.iteration

    def convergence_check(self):
        """
        Check whether convergence criterion is satisfied by comparing
        the relative error of the current estimate and the target density matrices.
        Utilized in "step" function below.
        """
        if (
            self.process_idx == 0
            and self.iteration % self.convergence_check_period == 0
        ):
            # compute relative error

            XkErrRatio = LA.norm(self.Xk - self.Xk_old) / LA.norm(self.Xk_old)

            xx = LA.norm(Col_flip_sign(self.uk) - Col_flip_sign(self.uk_old))

            ukErrRatio = xx / LA.norm(self.uk_old)

            print("            min(uk +/- uk_old)   = {}\n".format(xx))

            # self.Rec_st_Err.append( LA.norm(self.uk - self.uk_old) )
            self.Err_relative_Xk.append(XkErrRatio)

            self.Err_relative_st.append(ukErrRatio)

            if (
                min(ukErrRatio, XkErrRatio) <= self.relative_error_tolerance
            ):  #  using state or Xk to check convergence
                self.converged = True
                self.convergence_iteration = self.iteration
                print(
                    " *********  XkErrRatio = {} < StpTol   ******".format(XkErrRatio)
                )

            if self.target_DM is not None:
                Fro_diff = LA.norm(self.Xk - self.target_DM, "fro")

                if np.isnan(Fro_diff):
                    print(
                        " ********* {}-th Fro_diff = NaN = {}".format(
                            self.iteration, Fro_diff
                        )
                    )
                    # break
                    return

                fidelity_DM = state_fidelity(self.Xk, self.target_DM, validate=False)

                self.Target_Err_Xk.append(Fro_diff)
                self.fidelity_Xk_list.append(fidelity_DM)

                print(
                    "         relative_errU = {},  relative_errX = {}".format(
                        ukErrRatio, XkErrRatio
                    )
                )
                print("         Target_Error = {}\n".format(Fro_diff))

                if Fro_diff > 1e2:
                    raise ValueError("  Target_Err_Xk  too big !!!  Stop and check  \n")


from methods_ParmBasic import BasicParameterInfo


class BasicWorkerRGD(LoopRGD):
    """
    Basic worker class. Implements MiFGD, which performs
    accelerated gradient descent on the factored space U
    (where UU^* = rho = density matrix)
    """

    def __init__(self, params_dict):
        """the initialization of setting up the parameters for the optimizer

        Args:
                params_dict (dict): dictionary of parameters
        """

        BasicParameterInfo.__init__(self, params_dict)

        LoopRGD.__init__(self)

    def A_dagger(self, ym):
        """the implementation of A^+ (A dagger) operator

        Args:
                ym (ndarray): the input vector as the coefficient array for the Pauli matrices

        Returns:
                ndarray: the matrix output of A_dagger operator
        """
        YY = np.zeros((self.num_elements, self.num_elements))

        for yP, proj in zip(ym, self.projector_list):
            YY += yP * proj.matrix

        return self.coef * YY


    def Hr_tangentWk(self, Alpha, uGv):
        """Hard thresholding Hr(.) of the intermediate matrix Jk
                according to the 4th step in the RGD algorithm in the paper,
                and update the SVD of the new predicted self.Xk = Hr(Jk)

        Args:
                Alpha (float): the step size in the 2nd  step of the RGD algorithm
                uGv (ndarray): uk @ Gk @ vk

        Returns:
                ndarray: (uk) updated left singular vectors of self.Xk
                ndarray: (sDiag) updated singular values of self.Xk
                ndarray: (vk) updated right singular vectors of self.Xk
        """

        Nr = self.Nr

        # ------------------------------------------------- #
        # 	re-scaling Alpha according to uGv  or sDiag		#
        # ------------------------------------------------- #
        max_scaling_time = 1
        for ii_sc in range(max_scaling_time):
            #print("    ************  {}-th scaling for the Alpha".format(ii_sc))

            D2s = Alpha * uGv + self.sDiag

            if self.EigV_positive == 1:  #  Hermitian  &   all EigVal > 0
                Y2 = Alpha * (self.Gu - self.uk @ uGv)

                q2, r2 = LA.qr(Y2)
                # print("k={}, (q2.shape, r2.shape) = ({}, {})".format(k, q2.shape, r2.shape))
                U2r = np.concatenate((self.uk[:, :Nr], q2[:, :Nr]), axis=1)
                # print("U2r shape = {}".format(U2r.shape))

                D2s = np.concatenate((D2s, r2[:Nr, :Nr].T.conj()), axis=1)
                # print(D2s.shape)

            else:  #  not Hermitian  OR   Eig-val  <  0,  i.e.  U  !=  V
                Y1 = Alpha * (self.Gu - self.vk @ uGv.T.conj())
                Y2 = Alpha * (self.Gv - self.uk @ uGv)

                q1, r1 = LA.qr(Y1)
                # print("k={}, (q1.shape, r1.shape) = ({}, {})".format(k, q1.shape, r1.shape))
                q2, r2 = LA.qr(Y2)
                # print("k={}, (q2.shape, r2.shape) = ({}, {})".format(k, q2.shape, r2.shape))
                U2r = np.concatenate((self.uk[:, :Nr], q2[:, :Nr]), axis=1)
                # print("U2r shape = {}".format(U2r.shape))
                V2r = np.concatenate((self.vk[:, :Nr], q1[:, :Nr]), axis=1)
                # print("V2r shape = {}".format(V2r.shape))

                D2s = np.concatenate((D2s, r1[:Nr, :Nr].T.conj()), axis=1)
                # print(D2s.shape)

            D2s = np.concatenate(
                (D2s, np.concatenate((r2[:Nr, :Nr], np.zeros((Nr, Nr))), axis=1)),
                axis=0,
            )
            # print(D2s.shape)

            print(
                "    Hr_tangentWk  -->  Remain_Hermitian = {}, EigV_positve = {}".format(
                    self.Remain_Hermitian, self.EigV_positive
                )
            )
            print(
                "          D2s - D2s.T.conj() = {}".format(LA.norm(D2s - D2s.T.conj()))
            )
            print(
                "          np.allclose(D2s, D2s.T.conj()) = {}\n".format(
                    np.allclose(D2s, D2s.T.conj())
                )
            )

            if np.allclose(D2s, D2s.T.conj()) == True:
                Mu2r, s2r, Mv2rh = LA.svd(D2s, full_matrices=True, hermitian=True)

            else:  #  not Hermitian
                Mu2r, s2r, Mv2rh = LA.svd(D2s, full_matrices=True)

            Mu2r_Nr = Mu2r[:, :Nr]
            Mv2r_Nr = Mv2rh[:Nr, :].T.conj()

            #print("        s2r       = {}".format(s2r))
            #print("        max(s2r)  = {}".format(max(s2r)))
            if max(s2r) < 1:
                break
            else:
                Alpha = Alpha * 0.5

        # --------------------------------------------------------- #
        # 	END of  re-scaling Alpha according to uGv  or sDiag		#
        # --------------------------------------------------------- #

        self.sDiag = np.diag(s2r[:Nr])

        if self.EigV_positive == 1:  #  original  U = V

            uk = U2r.dot(Mu2r_Nr)
            vk = U2r.dot(Mv2r_Nr)  #  since may   Mu2r_Nr  !=  Mv2r_Nr

        else:  #  original  U  !=  V

            uk = U2r.dot(Mu2r_Nr)
            vk = V2r.dot(Mv2r_Nr)

        # ------------------------------------- #
        # 		update the Xk, uk, vk 			#
        # ------------------------------------- #

        self.check_Hermitian_x_y(uk, vk)

        self.uk = uk
        self.ukh = np.transpose(np.conj(uk))

        if self.EigV_positive == 1:  #  U  ==  V
            self.vkh = np.transpose(np.conj(uk))
            self.vk = uk

        else:
            self.vkh = np.transpose(np.conj(vk))
            self.vk = vk

        self.Xk = self.uk.dot(self.sDiag) @ self.vkh


    def calc_PtG_2_uG_Gv(self):
        """to project Gk onto the tangnet space and
                calculate the related objects

        Returns:
                ndarray: (uGv) uk@ Gk @ vk
        """

        Gu = np.zeros((self.num_elements, self.Nr), dtype=complex)

        if self.EigV_positive == 1:  # Hermitian &  EigV > 0  -->   U = V

            for proj, zmP in zip(*[self.projector_list, self.zm]):

                projU = proj.dot(self.uk)
                Gu += zmP * projU

            self.Gu = self.coef * Gu
            uGv = self.ukh @ self.Gu

            self.PtGk = (
                self.uk @ self.Gu.T.conj() + self.Gu @ self.ukh - self.uk @ uGv @ self.ukh
            )

        else:  #  U  !=  V

            Gv = np.zeros(self.vk.shape, dtype=complex)

            for zmP, proj in zip(self.zm, self.projector_list):

                projU = proj.dot(self.uk)
                projV = proj.dot(self.vk)
                Gu += zmP * projU
                Gv += zmP * projV

            print(
                " np.allclose( Col_flip_sign(Gu), Col_flip_sign(Gv) ) = {}".format(
                    np.allclose(Col_flip_sign(Gu), Col_flip_sign(Gv))
                )
            )

            self.Gv = self.coef * Gv
            self.Gu = self.coef * Gu
            uGv = self.ukh @ self.Gv

            self.PtGk = (
                self.uk @ self.Gu.T.conj()
                + self.Gv @ self.vkh
                - self.uk @ uGv @ self.vkh
            )

        return uGv

    def calc_n1_n2(self):
        # ----------------------------- #
        # 		calc AptGk = A(PtGk)	#
        # ----------------------------- #
        
        AptGk = self.coef * Amea(
            self.projector_list, self.PtGk
        )  # quickest

        n2 = LA.norm(AptGk)
        n1 = LA.norm(self.PtGk)

        # --------------------------------- #
        # 	calculate (n1, n2)				#
        # --------------------------------- #

        if n1 == 0.0:
            print("Projected Gradient PtGk norm = n1 = 0 = {}".format(n1))
            print("  -->  should achieve the minimum | or not successful ??")
            return

        Alpha = (n1 / n2) ** 2

        return Alpha


    def stepRGD(self):
        """each iteration step of the RGD algorithm"""

        self.Xk_old = self.Xk
        self.uk_old = self.uk

        print(" --------  {}-th iteration \n".format(self.iteration))
        tt1 = time.time()

        # ----------------------------------------- #
        # 	       calc of Axk = mapA(Xk)    		#
        # ----------------------------------------- #

        Axk = Amea(self.projector_list, self.Xk) * self.coef  ##  1st quickest
        self.zm = (
            self.yIni - Axk
        )  ## [yIni = coef*measurement_list] order follows self.label_list

        # ----------------------------------------------------- #
        # 	calculate (n1, n2)	by direct calculation of Gk		#
        # ----------------------------------------------------- #

        print(
            " {}-th step -> self.Remain_Hermitian = {}, self.EigV_positive = {}".format(
                self.iteration, self.Remain_Hermitian, self.EigV_positive
            )
        )

        uGv = self.calc_PtG_2_uG_Gv()

        Alpha = self.calc_n1_n2()

        self.Hr_tangentWk(Alpha, uGv)


        if self.iteration % 100 == 0:
            print("   ****  {}-th iteratation\n".format(self.iteration))

        tt7 = time.time()
        self.convergence_check()
        tt8 = time.time()

        print("     convergence check   -->  time = {}\n".format(tt8 - tt7))
        print("      stepRGD            -->  time = {}\n".format(tt8 - tt1))

        self.step_Time[self.iteration] = tt8 - tt1


############################################################
## Utility functions
## XXX To modularize/package
############################################################


# ========================================= #
# 	specifically for RGD					#
# ========================================= #


# -------------------------------------------------------------------- #
#       this one satisfy our unit convention                           #
# -------------------------------------------------------------------- #


def Amea(proj_list, XX):
    """the implementation of the sampling operator A

    Args:
            proj_list (list): list of Pauli matrix operators
            XX (ndarray): the input matrix
            m (int): = self.num_labels = the number of sampled Pauli matrices
            coef (float): the value of scaling for the final result

    Returns:
            ndarray: the final result of the sampling operator A
    """

    yMea = np.zeros(len(proj_list))     # len(proj_list) = the number of sampled Pauli matrices
    for ii, proj in enumerate(proj_list):
        ProjM = proj.matrix
        yMea[ii] = np.dot(
            ProjM.data, np.asarray(XX[ProjM.col, ProjM.row]).reshape(-1)
        ).real

    #return coef * yMea
    return yMea
    

def Col_flip_sign(xCol):
    """to change the sign of each column
            such that the largest element of each column is positive.

            The purpose of this function is for comparison, such as that will be
            used in check_Hermitian_x_y

    Args:
            xCol (ndarray): the input column vector

    Returns:
            ndarray: the update xCol with sign changed
    """

    if xCol.shape[0] < xCol.shape[1]:
        print("  ***   ERROR: This is a row, not a column   ***")
        return

    ID_max_abs = [np.argmax(np.abs(xCol[:, ii])) for ii in range(xCol.shape[1])]
    sign = np.sign([xCol[ID_max_abs[ii], ii] for ii in range(xCol.shape[1])])

    xCol = np.multiply(sign, xCol)

    return xCol


def power_Largest_EigV(M, criteria=1e-15, seed=0):
    """use the power method to obtain the largest eigenvalue and eigenvector

    Args:
            M (ndarray): the input matrix
            criteria (float, optional): stopping criteria of the power method.
                    Defaults to 1e-15.
            seed (int, optional): random seed parameter. Defaults to 0.

    if converged:
            Returns:
                    float: (lam_pseudo) the largest eigenvalue
                    ndarray: (Mv) the eigen vector of the largest eigenvalue
                    int: (ii) the number of iterations for convergence
    """

    converged = 0
    InitV = np.ones((M.shape[1], 1))

    for init_cnt in range(5):
        InitV = InitV / LA.norm(InitV)
        Vinput = InitV

        for ii in range(500):
            Mv = M @ Vinput
            lam_pseudo = LA.norm(Mv)
            Mv /= lam_pseudo

            Mv = Col_flip_sign(Mv)

            diff = LA.norm(Mv - Vinput)
            Vinput = Mv
            print(
                " {}-th iteration: lambda = {}, diff = {}".format(ii, lam_pseudo, diff)
            )
            if diff < criteria:
                break

        if diff > criteria:
            print(" *** power method not converged:  diff = {}".format(diff))
            print("       Now {}-th init  -->  try another InitV \n".format(init_cnt))

            InitV = np.random.RandomState().randn(M.shape[1], 1)

        else:
            print(
                " ****  Largest EigV  obtained from the power method:  ||Mv- v|| = {}  ***".format(
                    diff
                )
            )
            print(
                "               (at the {}-th initV; {}-th iteration)  \n".format(
                    init_cnt, ii
                )
            )
            converged = 1
            break

    if converged == 0:
        return
    elif converged == 1:
        return lam_pseudo, Mv, ii


def Hr_Hermitian(X, r, Choice=0):
    """hard theresholding the matrix X to rank r approximation

    Args:
            X (ndarray): the input matrix
            r (int): the target rank for the final approximated matrix
            Choice (int, optional): the method of doing SVD. Defaults to 0.

    Returns:
            ndarray: (u) the final left singular vector
            ndarray: np.diag(s)[:r, :r] = the matrix with diagonal elements the singular values
            ndarray: (vh) the complex conjugate of the right singular vector
    """

    print("  Choice for Hr_Hermitian = {}\n".format(Choice))

    if Choice == 0:
        u, s, vh = LA.svd(X, hermitian=True)

        u = np.array(u[:, :r])  #  only keep the first r columns
        vh = np.array(vh[:r, :])  #  only keep the first r rows
    elif Choice == 1:
        u, s, vh = svds(X, k=r)

    elif Choice == 2:  #  only for Nr = r = 1
        s, u, ConvergeCnt = power_Largest_EigV(X)
        s = [s]
        u = np.array(u)
        vh = u.T.conj()

    return u, np.diag(s)[:r, :r], vh

