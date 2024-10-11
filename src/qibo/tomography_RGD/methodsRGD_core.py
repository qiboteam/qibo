

import numpy as np
from numpy import linalg as LA
from scipy.sparse.linalg import svds

from functools import reduce

#from mpi4py import MPI
from qiskit.quantum_info import state_fidelity

import projectors
#import measurements


import scipy.sparse as sparse

import pickle
# --------------------------------- #
#	the origianl RGD package		#
# --------------------------------- #

#import scipy.sparse as sparse
import time

############################################################
## Basic sequential RGD
## XXX WIP
############################################################

class LoopRGD:
	def __init__(self):
		self.method = 'RGD'


	def initialize_RndSt(self):
		'''
		Random initialization of the state density matrix.
		'''

		print('\n ******  initial choice same as the MiFGD  ***** \n')
		print('self.seed = {}'.format(self.seed))
		self.InitChoice = 'random initial (same as MiFGD)'

		seed = self.seed

		stateIni = []
		for xx in range(self.Nr):
			real_state = np.random.RandomState(seed).randn(self.num_elements)

			#seed += 1		
			imag_state = np.random.RandomState(seed).randn(self.num_elements)

			seed += 1

			#real_state = np.random.RandomState().randn(self.num_elements)
			#imag_state = np.random.RandomState().randn(self.num_elements)

			state      = real_state + 1.0j * imag_state
		
			#state      = 1.0 / np.sqrt(self.num_tomography_labels) * state
			state_norm = np.linalg.norm(state)
			state      = state / state_norm

			stateIni.append(state)

		stateIni = np.array(stateIni).T

		self.state = stateIni
		self.uk = stateIni
		self.vk = stateIni
		self.ukh = stateIni.T.conj()
		self.vkh = np.transpose(stateIni.conj())

		self.s0_choice = 0
		if self.s0_choice == 0:
			#ss = np.random.random(self.Nr)
			#ss = ss / np.sum(ss)
			ss = np.ones(self.Nr)

		elif self.s0_choice == 1:
			ss = np.ones(self.Nr)  /   self.Nr


		self.sDiag = np.diag(ss)

		X0 = stateIni @  self.sDiag  @  self.vkh

		self.u0    = stateIni
		self.v0    = stateIni
		self.s0    = np.diag(ss)
		self.X0    = X0

		self.Xk    = X0

		self.check_Hermitian_x_y(self.uk, self.vk)


	def initialize_yAd(self):
		""" initialize the initial density matrix for the algorithm
		"""
		
		#print('\n          ******  initial choice following paper choice  ***** ')
		#self.Id = np.eye(self.num_elements, dtype=complex)  ##  matrix dimension d = num_elements = 2**Nk
		self.InitChoice = 'paper'

        # ------------------------------------------------------------ #

		if self.Ch_svd >= 0:					#   need to compute X0

			print('  start running self.A_dagger')
			tt1 = time.time()
			#y = self.yIni 	   ##  this y = coef * measurement_list -> order follows self.label_list

			if self.Md_tr == 0:		#  the normal def of A, i.e. sampling all Pauli
				Xk = self.A_dagger(self.yIni)					#  1st quickest

			elif self.Md_tr == 1 or self.Md_tr == 2:	#  add constraint on Identity
				Xk = self.A_dagger_tr1(self.projector_list, self.yIni, self.yI0)

			tt2 = time.time()

			print(' self.A_dagger done  -->  time = {}'.format(tt2-tt1))
			print(' Initial SVD -->  choice = {}'.format(self.Ch_svd))

			uk, sDiag, vkh = Hr_Hermitian(Xk, self.Nr, self.Ch_svd)		#  Ch_svd =0: full SVD

		elif self.Ch_svd < 0:					#   don't need to compute X0

			print(' *********   using randomized-SVD  to construct  X0 = uk @ sDiag @ vkh  **************************')

			tt2a = time.time()
			uk, sDiag, vkh = self.rSVD(k=self.Nr, s=2, p=15)
			tt2b = time.time()
			print('   self.rSVD           -->   time = {}'.format(tt2b-tt2a))

			#Compare_2_SVD(u1, sDiag1, v1h, uk, sDiag, vkh)


		self.check_Hermitian_x_y(uk, vkh.T.conj())

		if self.EigV_positive  == 1:		#   U == V
			vk  = uk
			vkh = uk.T.conj()

		else:								#   U != V
			vk  = vkh.T.conj()

		ukh = uk.T.conj() 
		Xk = uk @ sDiag @ vkh                   	# only keep rank r singular vectors


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


	def A_dagger_vec(self, vec):
		""" to obtain A^+(y) @ vec, where y is the input vector for A^dagger operator
			recorded in self.yIni

		Args:
			vec (ndarray): a vector which is supposed to be the left singular vector

		Returns:
			ndarray: Xu = the A^+(y) @ vec
		"""

		Xu  = np.zeros((self.num_elements, vec.shape[1]), dtype=complex)
		for yP, proj in zip(self.yIni, self.projector_list):
			Xu += yP * proj.dot(vec)
		Xu *= self.coef

		return Xu

	def A_dagger_ym_vec(self, zm, vec):
		"""  to obtain A^+(zm) @ vec

		Args:
			zm (ndarray): the input vector for the A^+ operator
			vec (ndarray): a vector which is supposed to be the left singular vector

		Returns:
			ndarray: Xu = the A^+(y) @ vec
		"""

		Xu  = np.zeros((self.num_elements, vec.shape[1]), dtype=complex)
		for yP, proj in zip(zm, self.projector_list):
			Xu += yP * proj.dot(vec)
		#Xu *= self.coef

		return Xu


	def rSVD(self, k, s=3, p=12, matrixM_Hermitian=1):
		""" randomized SVD

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

		qDs = min(k + s, self.num_elements)    # k + s small dimension for Q 

		Omega =np.random.RandomState().randn(self.num_elements, qDs)

		#Mw = M @ Omega
		#Mw = self.A_dagger_vec(Omega)
		Mw = self.A_dagger_ym_vec(self.yIni, Omega)

		Q, R = LA.qr(Mw)

		for ii in range(p):
			tt1 = time.time()

			ATQ = self.A_dagger_ym_vec(self.yIni, Q)

			G, R = LA.qr(ATQ)
			tt2 = time.time()

			AG   = self.A_dagger_ym_vec(self.yIni, G)

			Q, R = LA.qr(AG)        

			tt3 = time.time()
			print('  ***   {}-th (A A*) done:   Time --> ATQ: {},  AG: {}   ***'.format(ii, tt2-tt1, tt3-tt2))

		
		if matrixM_Hermitian == 0:
			#B = Q.T.conj() @ M
			B = self.A_dagger_vec(Q).T.conj()

			uB, s, vh = LA.svd(B)
			u  = Q @ uB[:, :k]              #  k = Nr
			s  = s[:k]                      #  k = Nr
			vh = vh[:k, :]                  #  k = Nr

		elif matrixM_Hermitian == 1:
			#B = Q.T.conj() @ M @  Q
			B = Q.T.conj() @ self.A_dagger_vec(Q)

			uB, s, vh = LA.svd(B, hermitian=True)
			u  = Q @ uB[:, :k]              #  k = Nr
			s  = s[:k]                      #  k = Nr

			vh = vh[:k, :] @ Q.T.conj()     #  k = Nr


		return u, np.diag(s), vh



	def check_Hermitian_x_y(self, x, y, Hermitian_criteria=1e-13):
		""" to check wheter x and y are the same,
			where x and y are supposed to be the left and right singular vectors
			if x = y, then the original SVDed matrix is Hermitian

		Args:
			x (ndarray): the 1st vector to compare, supposed to be the left singular vector
			y (ndarray): the 2nd vector to compare, supposed to be the right singular vector
			Hermitian_criteria (float, optional): the numerical error minimum criteria. Defaults to 1e-13.
		"""

		#if LA.norm(x - y) < Hermitian_criteria:
		if np.allclose(x, y):

			self.Remain_Hermitian = 1						#	True  for remaining Hermitian
			self.EigV_positive    = 1

		#elif LA.norm(Col_flip_sign(x) - Col_flip_sign(y)) < Hermitian_criteria:
		elif np.allclose( Col_flip_sign(x), Col_flip_sign(y)):
			self.Remain_Hermitian = 1
			self.EigV_positive    = 0

		else:
			self.EigV_positive    = None
			self.Remain_Hermitian = 0						#	False for remaining Hermitian
			self.st_Non_Hermitian.append(self.iteration)	#   start num for non-Hermitian

		self.EigV_pm_List.append(self.EigV_positive)
		self.Hermitian_List.append(self.Remain_Hermitian)

		print('            Hermitian ?    Remain_Hermiain = {},    EigV_postive = {}'.format(self.Remain_Hermitian, self.EigV_positive))


	def Get_measured_y(self, Md_tr=0):
		""" to get the coefficients of all sampled Pauli operators for the A^+ operator

		Args:
			Md_tr (int, optional): Method if including trace = 1 or not. Defaults to 0.
		"""

		if Md_tr==0:
			coef = np.sqrt(self.num_elements/self.num_labels)
			self.yIni = np.array(self.measurement_list) * coef
			self.coef = coef

		elif Md_tr == 1 or Md_tr == 2:
			Iden = ''.join(['I' for i in range(self.n)])
			self.label_list = self.label_list + [Iden]		#  No effect in compute

			projI = projectors.Projector(Iden)
			self.projI = projI

			dd = self.num_elements
			coef = np.sqrt((dd**2-1)/(dd*self.num_labels))

			self.yml_sc = coef       #  scale for ym from measurement_list
			self.yIni   = np.array(self.measurement_list) * coef
			self.yI0    = np.sqrt(1/self.num_elements)

		if Md_tr == 2:							# weighted sum
			self.w0 = 2
			self.wm = np.ones(self.num_labels)
			self.wm_sqrt = np.ones(self.num_labels)			#  each element is squred rooted

	def Load_Init(self, Ld):
		""" to load state vector decomposition (U, s, V) from file
			for the RGD optimization initialization

		Args:
			Ld (str): specification how to load the data from file

		"""

		F_X0 = '{}/X0.pickle'.format(self.meas_path)
		with open(F_X0, 'rb') as f:
			w_X0 = pickle.load(f)

		if Ld == 'RGD':
			u0, v0, sDiag, X0 = w_X0[Ld]

		else:
			U0, X0 = w_X0[Ld]			#  from 'MiFGD'

			self.Xk = X0
			if self.Nr == 1:
				self.uk    = U0
				self.vk    = U0
				self.ukh   = U0.T.conj()
				self.vkh   = U0.T.conj()
				self.sDiag = np.array([[1.0]])

			else:
				print('  Not Implemented Yet')

			self.check_Hermitian_x_y(self.uk, self.vk)

		self.X0 = X0


	def computeRGD(self, InitX, Ch_svd=0, Md_tr=0, Md_alp = 0, Md_sig=0, Ld=0):
		"""	Basic workflow of gradient descent iteration.

		1. initializes state dnesity matrix.
		2. mapkes a step (defined differently for each "Worker" class below) until 
		   convergence criterion is satisfied. 

		Args:
			InitX (int): choice of the initial matrix
			Ch_svd (int, optional): _description_. Defaults to 0.
			Md_tr (int, optional): Method if including trace = 1. Defaults to 0.
					Md_tr = 0,  the usual RGD algorithm, i.e. sampling all S
						  = 1,  to enforce coef for Identiy, i.e. not sampling Identity
			Md_alp (int, optional): method for scaling alpha. Defaults to 0.
			Md_sig (int, optional): method for scaling singular value. Defaults to 0.
			Ld (str, optional): specification how to load the data from file. Defaults to 0.
		"""
		#self.Id = np.eye(self.num_elements, dtype=complex)  ##  matrix dimension d = num_elements = 2**Nk

		self.InitX  = InitX
		self.Md_tr  = Md_tr						#  method of including identity operator or not

		self.Md_alp = Md_alp					#  method for scaling alpha
		self.Md_sig = Md_sig					#  method for scaling sinular values
		self.Ch_svd = Ch_svd					#  choice for initial SVD

		self.EigV_pm_List     = []				#  Eig-value is positive or not  (=1: positive,  = 0, non-positive, = None, no EigV)
		self.Hermitian_List   = []
		self.st_Non_Hermitian = []
		self.step_Time        = {}

		tt1 = time.time()

		self.iteration = -1

		self.Get_measured_y(Md_tr)
		print('  self.coef = {}'.format(self.coef))
		print('  InitX     = {}'.format(InitX))

		if InitX == 0:
			self.initialize_RndSt()				#  initial by random
		elif InitX == 1:
			self.initialize_yAd()				#  initial by A^\dagger(y)
		elif InitX == -1:
			self.Load_Init(Ld)


		tt2 = time.time()
		self.step_Time[-1] = tt2 - tt1

		print(' --------  X0  --> (-1)-th iteration  [ InitX = {}, SVD choice = {} ] \n'.format(InitX, Ch_svd))
		print('      X0 step (RGD)        -->  time = {}\n'.format(tt2 - tt1))


		X0_target_Err = LA.norm(self.Xk - self.target_DM, 'fro')
		self.Target_Err_Xk.append(X0_target_Err)		  	#  for X0
		self.InitErr = X0_target_Err

		print('      X0                   -->  Tr(X0)     = {}\n'.format(np.trace(self.X0)))		
		print('      X0                   -->  Target Err = {}\n'.format(X0_target_Err))		
		print('    RGD max num_iterations = {}\n'.format(self.num_iterations))


		self.uGv_list       = []
		self.Rec_Alpha      = []
		self.Alpha_sc		= []

		self.sDiag_list     = []
		self.zm_list        = []			#  = ym  - Axk
		self.z0_list        = []            #  = yI0 - A0


		for self.iteration in range(self.num_iterations):
			if not self.converged:
				self.stepRGD()
			else:
				break
		if self.convergence_iteration == 0:
			self.convergence_iteration = self.iteration

	
	def convergence_check(self):
		'''
		Check whether convergence criterion is satisfied by comparing
		the relative error of the current estimate and the target density matrices.
		Utilized in "step" function below.
		'''                
		if self.process_idx == 0 and self.iteration % self.convergence_check_period == 0:
			# compute relative error
			
			XkErrRatio = LA.norm(self.Xk - self.Xk_old) / LA.norm(self.Xk_old)

			xx =  LA.norm( Col_flip_sign(self.uk) - Col_flip_sign(self.uk_old))

			ukErrRatio = xx / LA.norm(self.uk_old)

			print('            min(uk +/- uk_old)   = {}\n'.format(xx))

			#self.Rec_st_Err.append( LA.norm(self.uk - self.uk_old) )
			self.Err_relative_Xk.append(XkErrRatio)

			self.Err_relative_st.append(ukErrRatio)


			if min(ukErrRatio, XkErrRatio) <= self.relative_error_tolerance:	#  using state or Xk to check convergence
				self.converged = True
				self.convergence_iteration = self.iteration
				print(' *********  XkErrRatio = {} < StpTol   ******'.format(XkErrRatio))


			if self.target_DM is not None:
				Fro_diff = LA.norm(self.Xk- self.target_DM, 'fro')

				if np.isnan(Fro_diff):
					print(' ********* {}-th Fro_diff = NaN = {}'.format(self.iteration, Fro_diff))
					#break
					return

				fidelity_DM = state_fidelity(self.Xk, self.target_DM, validate=False)

				self.Target_Err_Xk.append(Fro_diff)
				self.fidelity_Xk_list.append(fidelity_DM)

				print('         relative_errU = {},  relative_errX = {}'.format(ukErrRatio, XkErrRatio))
				print('         Target_Error = {}\n'.format(Fro_diff))

				if Fro_diff > 1e2:
					raise ValueError("  Target_Err_Xk  too big !!!  Stop and check  \n")


from methods_ParmBasic import BasicParameterInfo

class BasicWorkerRGD(LoopRGD):
	'''
	Basic worker class. Implements MiFGD, which performs
	accelerated gradient descent on the factored space U 
	(where UU^* = rho = density matrix)
	'''
	def __init__(self,
				 params_dict):
		""" the initialization of setting up the parameters for the optimizer

		Args:
			params_dict (dict): dictionary of parameters
		"""

		BasicParameterInfo.__init__(self, 
						params_dict)

		LoopRGD.__init__(self)


	def Amea_selfP(self, XX):
		""" do the sampling operator A(.) on the input matrix XX

		Args:
			XX (ndarray): the input matrix

		Returns:
			ndarray: the result of the sampling operator A(XX)
		"""

		yMea = np.zeros(self.num_labels)
		for ii, proj in enumerate(self.projector_list):
			ProjM = proj.matrix
			data  = ProjM.data
			col   = ProjM.col
			row   = ProjM.row
			yMea[ii] = np.dot(data, np.asarray(XX[col, row]).reshape(-1)).real
			
		return  self.coef * yMea
	
	def Amea(self, proj_list, XX):
		""" the implementation of the sampling operator A	

		Args:
			proj_list (list): list of Pauli matrix operators
			XX (ndarray): the input matrix

		Returns:
			ndarray: the final result of the sampling operator A
		"""

		yMea = [np.dot(proj_list[ii].matrix.data, \
		    np.asarray(XX[proj_list[ii].matrix.col, proj_list[ii].matrix.row]).reshape(-1)).real \
			for ii in range(len(proj_list))]  # correct
	
		return  np.sqrt(self.num_elements/self.num_labels) * np.array(yMea)


	def Amea_tr1(self, proj_list, XX):
		""" to get the coefficients of the sampled Pauli operator including the identity matrix
		in the basis expansion of density matrix XX

		Args:
			proj_list (list): list of all sampled Pauli operators
					supposed to be worker.projector_list
			XX (ndarray): matrix representing the density matrix

		Returns:
			ndarray: (yml) array representing the coefficients for each sampled Pauli operator
							in the basis expansion of the density matrix XX
			float  : (y0) coefficient of the identity in the basis expansion 
							of the density matrix XX
		"""

		yMea = [np.dot(proj_list[ii].matrix.data, \
		    np.asarray(XX[proj_list[ii].matrix.col, proj_list[ii].matrix.row]).reshape(-1)).real \
			for ii in range(self.num_labels)]  # correct
		yml = self.yml_sc * np.array(yMea)
		
		y0   = np.trace(XX).real / np.sqrt(self.num_elements)

		return  yml, y0


	def A_dagger(self, ym):
		""" the implementation of A^+ (A dagger) operator

		Args:
			ym (ndarray): the input vector as the coefficient array for the Pauli matrices

		Returns:
			ndarray: the matrix output of A_dagger operator
		"""
		YY = np.zeros((self.num_elements, self.num_elements))


		for yP, proj in zip(ym, self.projector_list):
			YY += yP * proj.matrix

		return  self.coef * YY

	def A_dagger_tr1(self, proj_list, ym, y0):
		""" the implementation of A^+ (A dagger) operator,
			similar to the function A_dagger 
			but with the identity matrix inherently included,
			i.e. adding the constraint for Identity 

		Args:
			proj_list (list): list of Pauli operator matrices
			ym (ndarray): the input vector as the coefficient array for the Pauli matrices
			y0 (float): the cofficient in front of the identity matrix

		Returns:
			ndarray: the matrix output of A_dagger operator
		"""
    	# fastest method
		YY = np.zeros(proj_list[0].matrix.shape)

		for ii in range(self.num_labels):
			YY += ym[ii]* proj_list[ii].matrix

		coef = self.yml_sc

		Y0 = self.projI.matrix * ( y0 / np.sqrt(self.num_elements) )

		return  coef * YY + Y0


	def Pt(self, Gk):
		""" projection to the tangent space spanned by uk and vk, where
			uk and vk are the left and right singular vectors of the current predction self.Xk

		Args:
			Gk (ndarray): a matrix to be projected onto the tangent space

		Returns:
			ndarray: (PtG) the projected matrix of the input matrix Gk onto the tangent space
			ndarray: (uGv) uk @ Gk @ vk
		"""

		uG  = self.ukh @ Gk
		Gv  = Gk @ self.vk
		uGv = uG @ self.vk

		PtG = self.uk @ uG + Gv @ self.vkh - self.uk @ (uGv @ self.vkh)
    
		self.Gv = Gv
		self.Gu = uG.T.conj()		#  suppose G is Hermitian

		return PtG, uGv


	def Hr_tangentWk(self, Alpha, uGv):
		""" Hard thresholding Hr(.) of the intermediate matrix Jk
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
		#	re-scaling Alpha according to uGv  or sDiag		#
		# ------------------------------------------------- #
		max_scaling_time = 1
		for ii_sc in range(max_scaling_time):
			print('    ************  {}-th scaling for the Alpha'.format(ii_sc))

			D2s = Alpha * uGv  + self.sDiag

			if self.EigV_positive  == 1:					#  Hermitian  &   all EigVal > 0
				Y2 = Alpha * (self.Gu - self.uk @ uGv)

				q2, r2 = LA.qr(Y2); #print("k={}, (q2.shape, r2.shape) = ({}, {})".format(k, q2.shape, r2.shape))
				U2r = np.concatenate((self.uk[:, :Nr], q2[:,:Nr]), axis=1); #print("U2r shape = {}".format(U2r.shape))

				D2s = np.concatenate((D2s, r2[:Nr, :Nr].T.conj()), axis=1); #print(D2s.shape)


			else:												#  not Hermitian  OR   Eig-val  <  0,  i.e.  U  !=  V
				Y1 = Alpha * (self.Gu - self.vk @ uGv.T.conj())
				Y2 = Alpha * (self.Gv - self.uk @ uGv)

				q1, r1 = LA.qr(Y1); #print("k={}, (q1.shape, r1.shape) = ({}, {})".format(k, q1.shape, r1.shape))
				q2, r2 = LA.qr(Y2); #print("k={}, (q2.shape, r2.shape) = ({}, {})".format(k, q2.shape, r2.shape))
				U2r = np.concatenate((self.uk[:, :Nr], q2[:,:Nr]), axis=1); #print("U2r shape = {}".format(U2r.shape))
				V2r = np.concatenate((self.vk[:, :Nr], q1[:,:Nr]), axis=1); #print("V2r shape = {}".format(V2r.shape))

				D2s = np.concatenate((D2s, r1[:Nr, :Nr].T.conj()), axis=1); #print(D2s.shape)


			D2s = np.concatenate((D2s, np.concatenate((r2[:Nr,:Nr], np.zeros((Nr,Nr))), axis=1)), axis=0); #print(D2s.shape)

			print('    Hr_tangentWk  -->  Remain_Hermitian = {}, EigV_positve = {}'.format(self.Remain_Hermitian, self.EigV_positive))
			print('          D2s - D2s.T.conj() = {}'.format(LA.norm( D2s - D2s.T.conj())))
			print('          np.allclose(D2s, D2s.T.conj()) = {}\n'.format(np.allclose(D2s, D2s.T.conj())))

			if np.allclose(D2s, D2s.T.conj()) == True:
				Mu2r, s2r, Mv2rh = LA.svd(D2s, full_matrices=True, hermitian=True)

			else:									#  not Hermitian   
				Mu2r, s2r, Mv2rh = LA.svd(D2s, full_matrices=True)


			Mu2r_Nr = Mu2r[:, :Nr]
			Mv2r_Nr = Mv2rh[:Nr, :].T.conj()

			print('        s2r       = {}'.format(s2r))
			print('        max(s2r)  = {}'.format(max(s2r)))
			if max(s2r) < 1:
				break
			else:
				Alpha = Alpha * 0.5

		self.Alpha_sc.append(ii_sc)
		self.Rec_Alpha.append(Alpha)
		print('    ------------  DONE svd of D2s with {}-th scaling (#max sc={})'.format(ii_sc, max_scaling_time))

		# --------------------------------------------------------- #
		#	END of  re-scaling Alpha according to uGv  or sDiag		#
		# --------------------------------------------------------- #

		sDiag = np.diag(s2r[:Nr])

		if self.EigV_positive == 1:					#  original  U = V
		
			uk = U2r.dot( Mu2r_Nr)
			vk = U2r.dot( Mv2r_Nr)					#  since may   Mu2r_Nr  !=  Mv2r_Nr

		else:										#  original  U  !=  V

			uk = U2r.dot( Mu2r_Nr )
			vk = V2r.dot( Mv2r_Nr )
			
		self.check_Hermitian_x_y(uk, vk)
		print('         error of Hermitian = LA.norm(|uk| - |vk|) = {}'.format(LA.norm(Col_flip_sign(uk) - Col_flip_sign(vk))))


		return uk, sDiag, vk

	def calc_GkV_UGk(self, zm):
		""" calculate the Gk, PtG, and uGv needed for the RGD algorithm

		Args:
			zm (ndarray): the input float array for the A^+
			   	corresponding to coefficients of all sampled Pauli operators		

		Returns:
			ndarray: (Gk) the matrix from the 1st step of the RGD algorithm	
			ndarray: (PtG) the Gk projected onto the tangent space
			ndarray: (uGv) uk @ Gk @ vk 
		"""

		YY = np.zeros(self.projector_list[0].matrix.shape)

		for ii in range(self.num_labels):
			YY += zm[ii]* self.projector_list[ii].matrix

		coef = np.sqrt(self.num_elements / self.num_labels)

		Gk = coef * YY

		uG  = self.ukh @ Gk
		Gv  = Gk @ self.vk
		uGv = uG @ self.vk

		PtG = self.uk @ uG + Gv @ self.vkh - self.uk @ (uGv @ self.vkh)
    
		self.Gv = Gv
		self.Gu = uG.T.conj()		#  suppose G is Hermitian

		return Gk, PtG, uGv


	def single_projUV_diff(self, measurement, proj):
			
		if self.EigV_positive == 1:					#   U  ==  V

			#Uproj   = proj.dot(self.uk).T.conj()
			projU    = proj.dot(self.uk)


			zmP = measurement - np.trace( self.sDiag @ self.vkh @ projU ).real		# quikcer

			zm_projU =  zmP *  projU

			return zm_projU

		else:										#   U   !=   V

			projU    = proj.dot(self.uk)

			projV  = proj.dot(self.vk)
			UprojV = self.ukh @ projV 	

			zm_projV =  zmP *  projV

			zm_projU =  zmP *  projU
			zm_UpV   =  zmP *  UprojV

			return zm_projU, zm_projV, zm_UpV


	def single_projUV_diff_zmP_UV(self, proj, zmP):
		""" to calculate zmP * proj @ uk
					and  zmP * proj @ vk

		Args:
			proj (class): the given Pauli operator object
			zmP (float): the coefficient corresponding to the given proj
				needed as input for the A^+ operator

		Returns:
			ndarray: (zm_projU) zmP * proj @ uk
			ndarray: (zm_projV) zmP * proj @ vk
		"""

		projU    = proj.dot(self.uk)
		projV    = proj.dot(self.vk)

		zm_projV =  zmP *  projV
		zm_projU =  zmP *  projU

		return zm_projU, zm_projV

	@staticmethod
	def single_projUV_diff_zmP_Hermitian(proj, uk):
		""" go obtain proj @ uk

		Args:
			proj (class): the given Pauli operator object
			uk (ndarray): the left singular vector of self.Xk

		Returns:
			ndarray: (projU) proj @ uk
		"""

		projU    = proj.dot(uk)

		return projU


	def calc_PtG_2_uG_Gv_Hermitian(self):
		""" to calculate uk @ Gk @ vk  
			and updated the projection of Gk onto the tangent space

		Returns:
			ndarray: (uGv) uk @ Gk @ vk 
		"""

		Gu  = np.zeros((self.num_elements, self.Nr), dtype=complex)

		for proj, zmP in zip(*[self.projector_list, self.zm]):

			projU = self.single_projUV_diff_zmP_Hermitian(proj, self.uk)
			Gu   +=  zmP * projU


		self.Gu = self.coef * Gu
		uGv     = self.ukh @ self.Gu

		self.PtGk = self.uk @ self.Gu.T.conj() +  self.Gu @ self.ukh  -  self.uk @ uGv @  self.ukh

		return uGv

	def calc_PtG_2_uG_Gv(self):
		""" to project Gk onto the tangnet space and
			calculate the related objects

		Returns:
			ndarray: (uGv) uk@ Gk @ vk
		"""


		Gu  = np.zeros((self.num_elements, self.Nr), dtype=complex)

		if self.EigV_positive  == 1:			#  U = V

			Have_Amea = 1
			if Have_Amea == 1:
				for proj, zmP in zip(*[self.projector_list, self.zm]):

					projU = self.single_projUV_diff_zmP_Hermitian(proj, self.uk)
					Gu   +=  zmP * projU

				self.Gu = self.coef * Gu
				uGv     = self.ukh @ self.Gu

				self.PtGk = self.uk @ self.Gu.T.conj() +  self.Gu @ self.ukh  -  self.uk @ uGv @  self.ukh


			elif Have_Amea == 0:

				for measurement, proj in zip(self.measurement_list, self.projector_list):

					zm_projU = self.single_projUV_diff(measurement, proj)

					Gu   +=  zm_projU

				self.Gu = (self.num_elements / self.num_labels) * Gu
				uGv     = self.ukh @ self.Gu

				self.PtGk = self.uk @ self.Gu.T.conj() +  self.Gu @ self.ukh  -  self.uk @ uGv @  self.ukh

		else:														#  U   !=   V
			Gv  = np.zeros(self.vk.shape, dtype=complex)

			for zmP, proj in zip(self.zm, self.projector_list):

				zm_projU, zm_projV = self.single_projUV_diff_zmP_UV(proj, zmP)

				Gu   +=  zm_projU
				Gv   +=  zm_projV

			print(' np.allclose( Col_flip_sign(Gu), Col_flip_sign(Gv) ) = {}'.format(np.allclose(Col_flip_sign(Gu), Col_flip_sign(Gv))))

			self.Gv = self.coef * Gv
			self.Gu = self.coef * Gu
			uGv     = self.ukh @  self.Gv
		
			self.PtGk = self.uk @ self.Gu.T.conj() +  self.Gv @ self.vkh  -  self.uk @ uGv @  self.vkh

		return uGv


	def calc_n1_n2_direct_Gk(self, zm, z0):
		""" calculate the necessary matrices in the RGD algorithm

		Args:
			zm (ndarray): coefficient array for the sampled Pauli matrices in the sampling operator A
			z0 (float): coefficient for the identify operator

		Returns:
			ndarray: (Gk) the matrix obtained in the 1st step of the RGD algorithm 
			ndarray: (uGv) u @ Gk @ v
			ndarray: (PtGk) Gk projected onto the tangent space
		"""
			
		# ------------------------------------------------- #
		#	  calc of Gk = A^+ (zm), where zm = y-Axk		#
		# ------------------------------------------------- #
		tt1 = time.time()

		if self.Md_tr == 0:
			Gk = self.A_dagger(zm)

		elif self.Md_tr == 1:
			Gk = self.A_dagger_tr1(self.projector_list, zm, z0)
		elif self.Md_tr == 2:
			zm = zm * self.wm			#  element-wise multiplication
			z0 = z0 * self.w0
			Gk = self.A_dagger_tr1(self.projector_list, zm, z0)

		tt2 = time.time()

		# --------------------------------- #
		#			PtGk = P(Gk)			#
		# --------------------------------- #

		PtGk, uGv = self.Pt(Gk)
	
		self.PtGk = PtGk

		tt3 = time.time()

		return Gk, uGv, PtGk


	def calc_n1_n2(self):
		# ----------------------------- #
		#		calc AptGk = A(PtGk)	#
		# ----------------------------- #
		tt1 = time.time()

		if self.Md_tr == 0:
			AptGk = Amea(self.projector_list, self.PtGk, self.num_labels, self.coef)	# quickest

			n2 = LA.norm(AptGk);    		#print(n2)

		elif self.Md_tr == 1:
			AptGk, A0 = self.Amea_tr1(self.projector_list, self.PtGk)
			n2 = np.sqrt(LA.norm(AptGk)**2 + A0**2)

		elif self.Md_tr == 2:
			AptGk, A0 = self.Amea_tr1(self.projector_list, self.PtGk)
			AptGk = AptGk * self.wm_sqrt

			n2 = np.sqrt(LA.norm(AptGk)**2 + A0**2 * self.w0)		# weighted sum in the inner product

		tt2 = time.time()

		n1        = LA.norm(self.PtGk);     		#print(n1)

		print('       [calc_n1_n2] AptGk     -->  time = {}\n'.format(tt2-tt1))

		# --------------------------------- #
		#	calculate (n1, n2)				#
		# --------------------------------- #
		
		if n1 == 0.0:
			print("Projected Gradient PtGk norm = n1 = 0 = {}".format(n1))
			print("  -->  should achieve the minimum | or not successful ??")
			return

		self.n1  = n1					#  not necessary to record
		self.n2  = n2					#  not necessary to record

		Alpha0 = (n1/n2)**2

		if self.Md_alp == 0:
			Alpha = Alpha0

		elif self.Md_alp == 1:
			Alpha = min(Alpha0, 1)

		elif self.Md_alp == 2:
			Alpha = min(Alpha0, 2)

		elif self.Md_alp == 3:
			Alpha = min(0.5*Alpha0, 0.8)

		return Alpha

	def zm_from_Amea(self):
		""" the sampling operator A(.)
		"""

		if self.Md_tr == 0:
			Axk = self.Amea_selfP(self.Xk)  ##  1st quickest 
			self.z0 = 0
		elif self.Md_tr == 1 or self.Md_tr == 2:
			Axk, A0 = self.Amea_tr1(self.projector_list, self.Xk)
		
			self.z0 = self.yI0 - A0
			self.z0_list.append(z0)

		self.zm = self.yIni - Axk	  ## [yIni = coef*measurement_list] order follows self.label_list

		self.zm_list.append(LA.norm(self.zm))


	def stepRGD(self):
		""" each iteration step of the RGD algorithm
		"""

		self.Xk_old = self.Xk
		self.uk_old = self.uk	

		print(' --------  {}-th iteration \n'.format(self.iteration))
		tt1 = time.time()

		# ----------------------------------------- #
		#	       calc of Axk = mapA(Xk)    		#
		# ----------------------------------------- #		

		self.zm_from_Amea()

		tt2a = time.time()

		# ----------------------------------------------------- #
		#	calculate (n1, n2)	by direct calculation of Gk		#
		# ----------------------------------------------------- #
		
		print(' {}-th step -> self.Remain_Hermitian = {}, self.EigV_positive = {}'.format(self.iteration, self.Remain_Hermitian, self.EigV_positive))


		if self.EigV_positive  == 1:		# Hermitian &  EigV > 0  -->   U = V

			uGv = self.calc_PtG_2_uG_Gv_Hermitian()

		else:								#  U  !=  V

			uGv = self.calc_PtG_2_uG_Gv()


		tp2b = time.time()

		print('       calc_PtG_2_uG_Gv       -->  time = {}'.format(tp2b - tt2a))


		Alpha = self.calc_n1_n2()

		uk, sDiag, vk = self.Hr_tangentWk(Alpha, uGv)

		# --------------------------------------------------------- #
		#	SOME approaches to scale the singular values or not		#
		# --------------------------------------------------------- #

		if self.Md_sig == 0:
			#ratio = 1
			self.sDiag = sDiag

		else:
			ss_Ary = np.diag(sDiag)

			if self.Md_sig == 1:
				ratio = 1/ np.sum(ss_Ary)							#  scaling of the sDiag
			elif self.Md_sig == 2:
				ratio = max(0.9, min(1/ np.sum(ss_Ary), 1.1))		#  scaling of the sDiag
			elif self.Md_sig == 3:
				if np.sum(ss_Ary) > 1:								#  similar to MiFGD case
					ratio = 1.0 / np.sum(ss_Ary)
				else:
					ratio = 1.0

			self.sDiag = ratio * sDiag


		# ------------------------------------- #
		#		update the Xk, uk, vk 			#
		# ------------------------------------- #
		self.uk  = uk
		self.ukh = np.transpose(np.conj(uk))

		if self.EigV_positive == 1:						#  U  ==  V
			self.vkh = np.transpose(np.conj(uk))
			self.vk  = uk
			
		else:
			self.vkh = np.transpose(np.conj(vk))
			self.vk  = vk

		self.Xk = self.uk.dot(self.sDiag) @ self.vkh

		self.uGv_list.append(uGv)

		self.sDiag_list.append(np.diag(self.sDiag))

		if self.iteration % 100 == 0:
			print('   ****  {}-th iteratation\n'.format(self.iteration))

		tt7 = time.time()
		self.convergence_check()
		tt8 = time.time()

		print('     convergence check   -->  time = {}\n'.format(tt8 - tt7))
		print('      stepRGD            -->  time = {}\n'.format(tt8 - tt1))

		self.step_Time[self.iteration] = tt8 - tt1

############################################################
## Utility functions
## XXX To modularize/package
############################################################

def density_matrix_norm(state):
	""" create density matrix from state

	Args:
		state (ndarray): state vector

	Returns:
		float: the norm of the density matrix 
	"""
	conj_state = state.conj()
	norm = np.sqrt(sum([v**2 for v in [np.linalg.norm(state * item)
									   for item in conj_state]]))
	return norm


def density_matrix_diff_norm(xstate, ystate):
	""" compare the difference between the density matrices constructed from xstate and ystate

	Args:
		xstate (ndarray): 1st state vector
		ystate (ndarray): 2nd state vector

	Returns:
		float: the norm of the matrix difference 
	"""
	conj_xstate = xstate.conj()
	conj_ystate = ystate.conj()
	
	norm = np.sqrt(sum([v**2 for v in [np.linalg.norm(xstate * xitem - ystate * yitem)
									   for xitem, yitem in zip(*[conj_xstate, conj_ystate])]]))
	return norm


# ========================================= #
#	specifically for RGD					#
# ========================================= #


# -------------------------------------------------------------------- #
#       this one satisfy our unit convention                           #
# -------------------------------------------------------------------- #

def Amea(proj_list, XX, m, coef):
	""" the implementation of the sampling operator A	

	Args:
		proj_list (list): list of Pauli matrix operators
		XX (ndarray): the input matrix
		m (int): = self.num_labels = the number of sampled Pauli matrices
		coef (float): the value of scaling for the final result

	Returns:
		ndarray: the final result of the sampling operator A
	"""
	
	yMea = np.zeros(m)
	for ii, proj in enumerate(proj_list):
		ProjM = proj.matrix
		yMea[ii] = np.dot(ProjM.data, np.asarray(XX[ProjM.col, ProjM.row]).reshape(-1)).real

	return coef * yMea


def A_dagger(proj_list, ym, m, Nk):
	""" the implementation of A^+ (A dagger) operator

	Args:
		proj_list (list): list of Pauli operator matrices
		ym (ndarray): the input vector as the coefficient array for the Pauli matrices
		m (int): the number of sampled Pauli operators
		Nk (int): the value of qubit number

	Returns:
		ndarray: the matrix output of A_dagger operator
	"""

	method = 0

    # fastest method
	if method == 0:
		YY = np.zeros(proj_list[0].matrix.shape)

		for ii in range(m):
			YY += ym[ii]* proj_list[ii].matrix

	return  np.sqrt(2 ** Nk) / np.sqrt(m) * YY


def Compare_2_SVD(u1, s1, v1h, u2, s2, v2h): 
	""" compare two SVD results and see if they are equal or not

	Args:
		u1 (ndarray): the left singular vector of the 1st SVD result
		s1 (ndarray): the diagnoal matrix of the singular values of the 1st SVD result
		v1h (ndarray): the complex conjugate of the right singular vector of the 1st SVD result
		u2 (ndarray): the left singular vector of the 2nd SVD result
		s2 (ndarray): the diagnoal matrix of the singular values of the 2nd SVD result
		v2h (ndarray): the complex conjugate of the right singular vector of the 2nd SVD result
	"""

	print('    s1 = {}, s2 = {}, diff = {}'.format(s1, s2, s1-s2))
	print('    np.allclose(u1,  u2)  = {}'.format(np.allclose(Col_flip_sign(u1), Col_flip_sign(u2))))

	print('    np.allclose(vkh, v5h) = {}'.format(np.allclose(Row_flip_sign(v1h), Row_flip_sign(v2h))))
	print('    np.allclose(vk,  v5)  = {}'.format(np.allclose(Col_flip_sign(v1h.T.conj()),  Col_flip_sign(v2h.T.conj()))))

	print('    np.allclose(xR1,  xR2)  = {}'.format(np.allclose( u1 @ s1 @ v1h,  u2 @ s2 @ v2h )))

	print('   u1  - u2  = {}'.format( LA.norm(u1  - u2)))
	print('   v1h - v2h = {}'.format( LA.norm(v1h - v2h)))
	print('   XR1 - XR2 = {}'.format( LA.norm(u1 @ s1 @ v1h - u2 @ s2 @ v2h)))
	print(' --------------------------------------------------------------------- ')


def rSVD(M, k, s=5, p=5):
    """         randomized SVD

        k   = Nr        # rank parameter
        s   = 5         # oversampling
        p   = 5         # power parameter

        ref:  https://arxiv.org/pdf/1810.06860.pdf    -->  A ~ QQ* A          (B = Q* A  )
			with modification:  if Remain_Hermitian   -->  A ~ QQ* A  QQ*     (B = Q* A Q)
    """
    qDs = min(k + s, M.shape[1])    # k + s small dimension for Q 

    Omega =np.random.RandomState().randn(M.shape[1], qDs)

    Mw = M @ Omega
    Q, R = LA.qr(Mw)

    for ii in range(p):
        ATQ  = M.T.conj() @ Q
        G, R = LA.qr(ATQ)

        AG   = M @ G
        Q, R = LA.qr(AG)        

        print('  ******    {}-th ATQ & AG done    *****'.format(ii))

    Remain_Hermitian = 1
    if Remain_Hermitian == 0:
        B = Q.T.conj() @ M

        uB, s, vh = LA.svd(B)
        u  = Q @ uB[:, :k]              #  k = Nr
        s  = s[:k]                      #  k = Nr
        vh = vh[:k, :]                  #  k = Nr

    elif Remain_Hermitian == 1:
        B = Q.T.conj() @ M @  Q

        uB, s, vh = LA.svd(B, hermitian=True)
        u  = Q @ uB[:, :k]              #  k = Nr
        s  = s[:k]                      #  k = Nr

        vh = vh[:k, :] @ Q.T.conj()     #  k = Nr


    return u, np.diag(s), vh


def Row_flip_sign(yRow):
	""" to change the sign of each row
		such that the largest element of each row is positive.
		
	Args:
		yRow (ndarray): the input row vector

	Returns:
		ndarray: the updated yRow vector
	"""

	if yRow.shape[0] >  yRow.shape[1]:
		print('  ***   ERROR: This is a colum, not a row   ***')
		return 
    
	ID_max_abs1 = np.argmax(np.abs( yRow), axis=1)
	ID_max_abs2 = [np.argmax(np.abs( yRow[ii, :]))  for ii in range(yRow.shape[0])]
	sign = np.sign([yRow[ii, ID_max_abs1[ii]] for ii in range(yRow.shape[0])])

	yRow = [ sn*yr for sn, yr in zip(sign, yRow)]
	yRow = np.array(yRow)

	return yRow



def Col_flip_sign(xCol):
	""" to change the sign of each column 
		such that the largest element of each column is positive.
	
		The purpose of this function is for comparison, such as that will be 
		used in check_Hermitian_x_y

	Args:
		xCol (ndarray): the input column vector
	
	Returns:
		ndarray: the update xCol with sign changed
	"""

	if xCol.shape[0] <  xCol.shape[1]:
		print('  ***   ERROR: This is a row, not a column   ***')
		return 

	ID_max_abs = [np.argmax(np.abs( xCol[:, ii]))  for ii in range(xCol.shape[1])]
	sign = np.sign([xCol[ID_max_abs[ii], ii] for ii in range(xCol.shape[1])])
	
	xCol = np.multiply(sign, xCol)

	return xCol


def power_Largest_EigV(M, criteria = 1e-15, seed=0):
	""" use the power method to obtain the largest eigenvalue and eigenvector

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
	InitV  = np.ones((M.shape[1], 1))

	for init_cnt in range(5):
		InitV  = InitV / LA.norm(InitV)
		Vinput = InitV

		for ii in range(500):
			Mv = M @ Vinput
			lam_pseudo = LA.norm(Mv)        
			Mv /= lam_pseudo
            
			Mv = Col_flip_sign(Mv)

			diff   = LA.norm( Mv - Vinput)
			Vinput = Mv 
			print(' {}-th iteration: lambda = {}, diff = {}'.format(ii, lam_pseudo, diff))
			if diff < criteria:
				break

		if diff > criteria:
			print(' *** power method not converged:  diff = {}'.format(diff))
			print('       Now {}-th init  -->  try another InitV \n'.format(init_cnt))

			InitV  = np.random.RandomState().randn(M.shape[1], 1)


		else:
			print(' ****  Largest EigV  obtained from the power method:  ||Mv- v|| = {}  ***'.format(diff))
			print('               (at the {}-th initV; {}-th iteration)  \n'.format(init_cnt, ii))
			converged = 1
			break

	if converged == 0:
		return
	elif converged == 1:
		return lam_pseudo, Mv, ii


def Hr_Hermitian(X, r, Choice=0):
	""" hard theresholding the matrix X to rank r approximation

	Args:
		X (ndarray): the input matrix
		r (int): the target rank for the final approximated matrix 
		Choice (int, optional): the method of doing SVD. Defaults to 0.

	Returns:
		ndarray: (u) the final left singular vector
		ndarray: np.diag(s)[:r, :r] = the matrix with diagonal elements the singular values
		ndarray: (vh) the complex conjugate of the right singular vector
	"""

	print('  Choice for Hr_Hermitian = {}\n'.format(Choice))

	if Choice == 0:
		u, s, vh = LA.svd(X, hermitian=True)

		u  = np.array(u[:, :r])		#  only keep the first r columns
		vh = np.array(vh[:r, :])		#  only keep the first r rows
	elif Choice == 1:
		u, s, vh = svds(X, k=r)

	elif Choice == 2:			#  only for Nr = r = 1
		s, u, ConvergeCnt = power_Largest_EigV(X)
		s = [s]
		u = np.array(u)
		vh = u.T.conj()

	return u, np.diag(s)[:r, :r], vh


def P(u, vh, Gk, rank):
	""" project the matrix Gk to the tangent space

	Args:
		u (ndarray): the left singular vector
		vh (ndarray): complex conjugate of the right singular vector
		Gk (ndarray): the input matrix
		rank (int): the rank of the singular value decomposed matrix

	Returns:
		ndarray: the matrix Gk projected to the tangent space
	"""
	#Gk = Gk

	u = u[:, :rank]
	v = np.transpose(np.conj(vh))

	v = v[:, :rank]

	U = np.matmul(u, np.transpose(np.conj(u)))
	V = np.matmul(v, np.transpose(np.conj(v)))

	return np.matmul(U, Gk) + np.matmul(Gk, V) - np.matmul(np.matmul(U, Gk), V)


def calc_n1_n2(zm, Ai, uk, vk, ukh, vkh, dd, m):

    # --------------------------------------------------------- #
    #   < method 2 >    go around Gk                            #
    # --------------------------------------------------------- #

    # ---------------------------------------------- #
    #   to construct a sequence of 
    #       Ei   =       Ai * V
    #       Pi   =       Ai * U
    #       Pi^+ = U^+ * Ai
    #       Fi   = U^+ * Ai * V
    # ---------------------------------------------- # 

    Pi = [Ai[kk] @ uk for kk in range(m)]	
    Ei = [Ai[kk] @ vk for kk in range(m)]
    Fi = [vkh @ Ei[kk] for kk in range(m)]

    zPi = [zm[ii] * Pi[ii] for ii in range(m)]
    zEi = [zm[ii] * Ei[ii] for ii in range(m)]
    zFi = [ukh @ zEi[ii] for ii in range(m)]    

    Eis   = [Ei[ii].T.conj() for ii in range(m)]    ## =  Ei^*  \in  (r, dd)
    Fis   = [Fi[ii].T.conj() for ii in range(m)]    ## =  Fi^* = V^* @ G @ U \in (r,r)

    fsum  = lambda x,y: x+y
    GkU   = reduce(fsum, zPi)   ## = Gk @ U   = sum over zi * Ai @ U  \in (dd, r)
    GkV   = reduce(fsum, zEi)   ## = Gk @ V   = sum over zi * Ai @ V  \in (dd, r)
    UsG   = GkU.T.conj()        ## = U^* @ Gk

    UsGV  = reduce(fsum, zFi)   ## = U^* @ Gk @ V = sum over zi * U^* @ Ai @ V \in (r,r)

    # --------------------------------------------- #
    #   calculating Y1t = PvPerp @ Gk @ worker.uk   #
    #   -->   need  zPi = zi * Ai @  worker.uk      #
    # --------------------------------------------- #

    Y1t = vkh @ GkU       ## \in (r, r) | GkU = xx = reduce(lambda x,y: x+y, zPi) \in (dd, r)
    Y1t = GkU - vk @ Y1t  ## \in (dd,r)

    # --------------------------------------------- #
    #   calculating Y2t = PuPerp @ Gk @ worker.vk   #
    #   -->   need  zEi = zi * Ai @ worker.vk       #
    # --------------------------------------------- #

    Y2t = ukh @ GkV       ##  \in (r, r) | GkV = xx = reduce(lambda x,y: x+y, zEi) \in (dd, r)
    Y2t = GkV - uk @ Y2t  ##  \in (dd,r)

    # ----------------------------------------------------- #
    #   uGv = worker.ukh @ Gk @ worker.vk   \in  C(r,r)     #
    #   -->   need  zFi  = zi * worker.ukh @ Ai @ worker.vk #
    # ----------------------------------------------------- #

    n1 = LA.norm(UsGV)**2 + LA.norm(Y1t)**2 + LA.norm(Y2t)**2    
    n1 = np.sqrt(dd/m) * np.sqrt(n1)
    
    Y1t = np.sqrt(dd/m) * Y1t
    Y2t = np.sqrt(dd/m) * Y2t

    # ---------------------------------------------------- #
    #   (II)    to calculate || A(PtGk) ||_2  = worker.n2  #
    #           where  dd = 2^Nk                           #
    # ---------------------------------------------------- #

    PzP = [np.trace( UsG @ Pi[ii]) for ii in range(m)]  ## mat \in (r,r)
    EzE = [np.trace(Eis[ii] @ GkV) for ii in range(m)]  ## mat \in (r, r)
    
    FzF1 = [np.trace(Fis[ii] @ UsGV) for ii in range(m)]  # mat \in (r, r)    
    AptGk = np.array(PzP) + np.array(EzE) - np.array(FzF1)

    
    AptGk = (dd/m) * np.array(AptGk).real


    n2 = LA.norm(AptGk)

    return n1, n2, UsGV*np.sqrt(dd/m), Y1t, Y2t

