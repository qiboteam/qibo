

import numpy as np
from numpy import linalg as LA
from scipy.linalg import interpolative as interp

#from mpi4py import MPI
from qiskit.quantum_info import state_fidelity

import time

import pickle

############################################################
## Basic sequential projFGD
## XXX WIP
############################################################

class LoopMiFGD:
	def __init__(self):
		self.method = 'MiFGD'


	def Rnd_stateU(self):
		''' Random initialization of the state density matrix.

		Returns:
			ndarray: the generated random initial U for the state density matrix
		'''

		print('self.seed = {}'.format(self.seed))

		seed = self.seed

		stateU = []

		for xx in range(self.Nr):
			real_state = np.random.RandomState(seed).randn(self.num_elements)
			
			imag_state = np.random.RandomState(seed).randn(self.num_elements)

			seed += 1

			#real_state = np.random.RandomState().randn(self.num_elements)
			#imag_state = np.random.RandomState().randn(self.num_elements)

			state      = real_state + 1.0j * imag_state

			#state      = 1.0 / np.sqrt(self.num_tomography_labels) * state
			state_norm = np.linalg.norm(state)
			state      = state / state_norm

			stateU.append(state)

		stateU = np.array(stateU).T

		self.U0 = stateU

		return stateU


	def est_Estimation_specNrm(self, stateU):
		""" estimate the spectral norm

		Args:
			stateU (ndarray): state vector representing a state
		"""
		ZZ0 =  stateU @ stateU.T.conj()
		nZ0_specN = interp.estimate_spectral_norm(ZZ0)

		Amea1 = np.zeros(self.num_labels)
		cnt   = np.arange(self.num_labels)
		for ii, projector, measurement in zip(*[cnt, self.projector_list, self.measurement_list]):
			projU = projector.dot(stateU)
			trAUU = np.trace(np.dot(stateU.T.conj(), projU)).real

			Amea1[ii] = trAUU - measurement

		AAz = self.A_dagger_InputZ(Amea1)
		nAA_specN = interp.estimate_spectral_norm(AAz)

		DeNom_SN = 11/10 * nZ0_specN + nAA_specN
		etaSpecN = 1/(4*DeNom_SN)

		self.AmeaZZ0 = Amea1 
		
		self.etaSpecN   = etaSpecN
		self.nZ0_specN  = nZ0_specN
		self.nAA_specN  = nAA_specN
		self.DeNom_SN   = DeNom_SN


	def eta_Estimation(self, stateU):
		ZZ0 =  stateU @ stateU.T.conj()
		nZ0 =  LA.norm(ZZ0)

		Amea1 = np.zeros(self.num_labels)
		#Amea2 = np.zeros(self.num_labels)
		cnt   = np.arange(self.num_labels)
		for ii, projector, measurement in zip(*[cnt, self.projector_list, self.measurement_list]):
			projU = projector.dot(stateU)
			trAUU = np.trace(np.dot(stateU.T.conj(), projU)).real

			Amea1[ii] = trAUU - measurement

		AAz = self.A_dagger_InputZ(Amea1)
		nAA = LA.norm(AAz)

		DeNom = 11/10 * nZ0 + nAA
		etaTh = 1/(4*DeNom)

		self.AmeaZZ0 = Amea1 
		
		self.etaTh = etaTh
		self.nZ0   = nZ0
		self.nAA   = nAA
		self.DeNom = DeNom


	def A_dagger_InputZ(self, InputZ):
		"""   A^+(y)
		"""
		YY = np.zeros((self.num_elements, self.num_elements))

		for zz, proj in zip(InputZ, self.projector_list):
			YY += zz * proj.matrix

		return YY


	def A_dagger_y(self):
		"""   A^+(y),  y is the measurement input
		"""
		YY = np.zeros((self.num_elements, self.num_elements))

		for yP, proj in zip(self.measurement_list, self.projector_list):
			YY += yP * proj.matrix

		return YY

	def ProjPSD(self, Rho):
		val, eigV = np.linalg.eig(Rho)

		if np.allclose(val.imag, 0):
			val = val.real
		
		#self.val  = val
		#self.eigV = eigV 

		#X0_reconstruct = eigV @ np.diag(val) @ eigV.T.conj()
		#if np.allclose(X0_reconstruct, self.X0):
		#	print(' the Hermitian decomposition is corect')
		#else:
		#	print(' sth wrong in the Hermitian decomposition')

		pos   = val > 0
		posID = np.arange(self.num_elements)[pos]

		# --------------  project to PSD  ------------------------- #
		Valpos = val[posID]			#  positive eigen value
		posEV  = eigV[:, posID]		#  corresponding eigV with val > 0

		sortID = np.argsort(Valpos)[::-1]	#  from large to small
		ID_Nr  = sortID[:self.Nr]			#  only keep the largest Nr elem

		Val_Nr = Valpos[ID_Nr]			#  Nr positive eig-values
		EV_Nr  = posEV[:, ID_Nr]		#  the corresponding Eig Vec

		#sqrVal = np.sqrt(Valpos)						#  keep all positive
		#U0     = posEV @  np.diag(sqrVal)				#  keep all positive
		#X1 = posEV @ np.diag(Valpos) @ posEV.T.conj()	#  keep all positive
		#X2 = U0 @ U0.T.conj()							#  keep all positive

		coef   = np.sqrt(10/11)							#  np.sqrt(1/L_hat)

		sqrVal = np.sqrt(Val_Nr)						#  keep only Nr elem
		
		U0     = EV_Nr @ np.diag(sqrVal*coef)			#  keep only Nr elem
		#X1 = EV_Nr @ np.diag(Val_Nr) @ EV_Nr.T.conj()	#  X1 should = X2
		#X2 = U0 @ U0.T.conj()							#  X1 should = X2

		#self.sortID = sortID		
		#self.ID_Nr  = ID_Nr

		self.Val_Nr = Val_Nr		#   Nr positive Eig Value
		self.Valpos = Valpos		#  all positive Eig Value
		#self.posEV  = posEV			#  all positive Eig vector
		#self.EV_Nr  = EV_Nr			#   Nr positive Eig vector
		self.U0     = np.array(U0)

		return np.array(U0)

	def initialize(self, stateU):

		self.stateU  = stateU
		self.momU    = stateU
		self.momUdag = stateU.T.conj()
		#self.XkU     =  stateU @ self.momUdag

		#print(' stateU  = \n{}'.format(stateU))
		#print(' momUdag = \n{}'.format(self.momUdag))

		#self.state = state
		#self.momentum_state = state[:]

		#self.Xk = np.outer(state, state.T.conj())		
		X0 = np.dot(stateU, stateU.T.conj())

		self.X0 = X0

		self.Xk = X0

		#print('real_state    = {}'.format(real_state))
		#print('imag_state    = {}'.format(imag_state))
		#print('num_tomography_labels   = {}'.format(self.num_tomography_labels))
		#print('state_norm    = {}'.format(state_norm))

	def Load_Init(self, Ld):
		""" to load state vector decomposition from file for MiFGD initialization

		Args:
			Ld (str): specification how to load the data from file

		Returns:
			ndarray: D * rank matrix representing state vector
		"""

		F_X0 = '{}/X0.pickle'.format(self.meas_path)
		print(' F_X0  = {}'.format(F_X0))
		print('  Ld   = {}'.format(Ld))

		with open(F_X0, 'rb') as f:
			w_X0 = pickle.load(f)

		#print(' w_X0  = {}'.format(w_X0))

		if Ld == 'RGD':
			u0, v0, s0, X0 = w_X0[Ld]

			stateU = u0
			for ii in range(self.Nr):
				scale = np.sqrt(s0[ii, ii])
				stateU[:, ii] = scale * u0[:, ii]

			X0_reconstruct = stateU @ stateU.T.conj()
			print(' X0 - X0_reconstruct = {}\n'.format(np.linalg.norm(X0 - X0_reconstruct)))

		else:
			U0, X0 = w_X0[Ld]
			stateU = U0

		return stateU		


	def compute(self, InitX=0, Ld=0):
		''' do the MiFGD optimization method
		
		Basic workflow of gradient descent iteration.
		1. randomly initializes state dnesity matrix.
		2. makes a step (defined differently for each "Worker" class below) until 
		   convergence criterion is satisfied. 

		Args:
			InitX (int, optional): choice of different initialization. Defaults to 0.
			Ld (str, optional): specification how to load the data from file. Defaults to 0.
  
		'''

		self.InitX            = InitX
		self.step_Time        = {}
		self.Init_Time        = {}

		tt1 = time.time()
		
		if InitX == 0:
			stateU      = self.Rnd_stateU()

		elif InitX == 1:
			Adgr_y  = self.A_dagger_y()
			stateU  = self.ProjPSD(Adgr_y)			

		elif InitX == -1:
			stateU = self.Load_Init(Ld)

		#print(' stateU = {}'.format(stateU))
		tt2 = time.time()

		if self.Option == 1:
			self.eta_Estimation(stateU)
			self.eta  =  self.etaTh
		elif self.Option == 2:
			self.est_Estimation_specNrm(stateU)
			self.eta  =  self.etaSpecN


		tt3 = time.time()
		self.initialize(stateU)

		tt4 = time.time()
		self.step_Time[-1] = tt4 - tt1
		
		self.Init_Time['X0']   = tt2 - tt1
		self.Init_Time['eta']  = tt3 - tt2
		self.Init_Time['momU'] = tt4 - tt3
		self.Init_Time['Tini'] = tt4 - tt1


		print(' --------  X0  --> (-1)-th iteration  [ InitX = {} ] \n'.format(InitX))
		print('      X0 step (MiFGD)      -->  time = {}\n'.format(tt4 - tt1))

		X0_target_Err = np.linalg.norm(self.Xk - self.target_DM, 'fro')
		self.Target_Err_Xk.append(X0_target_Err)			#  for X0
		self.InitErr = X0_target_Err

		print('      X0                   -->  Tr(X0)     = {}\n'.format(np.trace(self.X0)))		
		print('      X0                   -->  Target Err = {}\n'.format(X0_target_Err))
		print('    MiFGD num_iterations = {}\n'.format(self.num_iterations))

		self.Unorm =  []			#  to record the norm of state  U


		#print(' ****************************************** ')
		#print('self.InitX = {}'.format(self.InitX))
		#print('self.Nr  = {}'.format(self.Nr))
		#print('self.momU.shape = {}'.format(self.momU.shape))
		#print('self.momUdag.shape = {}'.format(self.momUdag.shape))
		#print('self.stateU.shape  = {}'.format(self.stateU.shape))
		#print('self.U0.shape = {}'.format(self.U0.shape))

		for self.iteration in range(self.num_iterations):
			if not self.converged:
				self.step()
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

			#numerator            = density_matrix_diff_norm(self.state, self.previous_state)
			#denominator          = density_matrix_norm(self.state)
			#error                = numerator  
			#relative_error       = numerator / denominator

			Run_state_err = 0
			if self.Nr == 1 and Run_state_err==1:
				#error                = density_matrix_diff_norm(self.state, self.previous_state)
				#relative_error       = error / density_matrix_norm(self.state)
				error                = density_matrix_diff_norm(self.stateU, self.previous_stateU)
				relative_error       = error / density_matrix_norm(self.stateU)

				self.error_list.append(error)

			#relative_errU  = np.linalg.norm(self.stateU - self.previous_stateU) / np.linalg.norm(self.previous_stateU)
			xx             = np.linalg.norm(Col_flip_sign(self.stateU) - Col_flip_sign(self.previous_stateU))
			relative_errU  = xx / np.linalg.norm(self.previous_stateU)


			#self.relative_error_list.append(relative_error)
			#self.Err_relative_st.append(relative_error)
			self.Err_relative_st.append(relative_errU)			

			XkErrRatio           = np.linalg.norm(self.Xk - self.Xk_old) / np.linalg.norm(self.Xk_old)
			self.Err_relative_Xk.append(XkErrRatio)

			#print(' || Xk - Xk(old) ||_F = {}'.format(np.linalg.norm(self.Xk - self.Xk_old)))
			#print(' | psi - pis(old) | = {}'.format(np.linalg.norm(self.stateU - self.previous_stateU)))

			if self.target_DM is not None:				# adding for checking the target error

				##  from  LA.norm(DM_Wst - rho.full())		need  two matrices

				#Xk = np.outer(self.state, self.state.T.conj())
				#self.Xk = Xk

				numerator2 = np.linalg.norm(self.Xk - self.target_DM)

				fidelity_DM = state_fidelity(self.Xk, self.target_DM, validate=False)

				self.Target_Err_Xk.append(numerator2)		
				self.fidelity_Xk_list.append(fidelity_DM)


				print('         relative_errU = {},  relative_errX = {}'.format(relative_errU, XkErrRatio))
				print('         Target_Error = {}\n'.format(numerator2))

			#if relative_error <= self.relative_error_tolerance:
			#if relative_errU <= self.relative_error_tolerance:
			if min(relative_errU, XkErrRatio) <= self.relative_error_tolerance:				
				self.converged = True
				self.convergence_iteration = self.iteration

			if self.target_state is not None:			
				
				##	from  density_matrix_diff_norm(Wst, target_St)  need two states
				#numerator1 = density_matrix_diff_norm(self.state, self.target_state)
				#self.FrobErr_st.append(numerator1)		#  =  self.target_error_list

				Xk_Err_compare = 0
				if Xk_Err_compare == 1:				
				#if self.Nr == 1:
					# ----------------- #				
					#  	original MiFGD	#
					# ----------------- #				

					# compute target relative error    (numerator extremely slow)
					numerator             = density_matrix_diff_norm(self.stateU, self.target_state)	# extremely slow
					denominator           = density_matrix_norm(self.target_state)

					target_error          = numerator
					target_relative_error = numerator / denominator
					self.target_error_list.append(target_error)		#  = self.Target_Err_Xk
					self.target_relative_error_list.append(target_relative_error)


				### check: if NOT validate=False then this fails
				fidelity = state_fidelity(self.target_state, self.stateU, validate=False)

				self.fidelity_list.append(fidelity)

#import methods_GetParam
#import methodsRGD
#from methodsRGD import WorkerRGD
#from methods_GetParam import WorkerParm
from methods_ParmBasic import BasicParameterInfo

#class BasicWorker(WorkerParm, LoopMiFGD):
class BasicWorker(BasicParameterInfo, LoopMiFGD):
	'''
	Basic worker class. Implements MiFGD, which performs
	accelerated gradient descent on the factored space U 
	(where UU^* = rho = density matrix)
	'''
	def __init__(self,
				 params_dict, params_MiFGD):
		""" initialization step for optimzation: to load all the sampled Pauli 
		    operators and measurement results

		Args:
			params_dict (dict): dictionary of parameters
		"""
		
		eta           = params_MiFGD['eta']		# for MiFGD		
		mu            = params_MiFGD.get('mu', 0.0)
		Option        = params_MiFGD.get('Option', 0)

		self.eta      = eta   		# step size / learning rate  (for MiFGD)
		self.mu       = mu   		# MiFGD momentum parameter (large: more acceleration)
		self.Option   = Option

		#WorkerParm.__init__(self,
		#				process_idx,
		#				num_processes,
#						params_dict, input_S)
		#				params_dict)

		BasicParameterInfo.__init__(self,
						params_dict)

		LoopMiFGD.__init__(self)

		
	@staticmethod
	def single_projection_diff(projector, measurement, state):
		'''
		Computes gradient of the L2-norm objective function.
		- Objective function: f(u) = 0.5 * \sum_i (<A_i, uu*> - y_i)^2
		- Gradient of f(u) wrt u: grad(u) = \sum_i (<A_i, uu*> - y_i) * A_i * u := \sum_i  grad_i (u)

		Explanation of the variables below:
		- projection: A_i * u term --> grad_i(u) = (trace(projection u*) - y_i) * projection
		- trace = np.dot(projection, state.conj())  = trace(projection u*)
		- diff = (trace - measurement) * projection = grad_i(u)
		'''
		projection = projector.dot(state)
		trace      = np.dot(projection, state.conj())
		diff       = (trace - measurement) * projection
		return diff


	@staticmethod
	def single_projU_diff(projector, measurement, stateU, Udag):
		'''
		Computes gradient of the L2-norm objective function.
		- Objective function: f(u) = 0.5 * \sum_i (<A_i, uu*> - y_i)^2
		- Gradient of f(u) wrt u: grad(u) = \sum_i (<A_i, uu*> - y_i) * A_i * u := \sum_i  grad_i (u)

		Explanation of the variables below:
		- projection: A_i * u term --> grad_i(u) = (trace(projection u*) - y_i) * projection
		- trace = np.dot(projection, state.conj())  = trace(projection u*)
		- diff = (trace - measurement) * projection = grad_i(u)

			stateU  =  self.momU,   i.e. the momentum state
		 	Udag    =  self.momU.T.conj() = self.momUdag
		'''
		#projection = projector.dot(state)
		#trace      = np.dot(projection, state.conj())
		#diff       = (trace - measurement) * projection

		projU  = projector.dot(stateU)
		#trAUU  = np.trace(np.dot(projU, Udag))
		trAUU  = np.trace(np.dot(Udag, projU)).real

		diffU  = (trAUU - measurement) * projU
	
		#print(' ------------------------ ')
		#print('projection = {}'.format(projection))
		#print('projection.shape = {}'.format(projection.shape))
		#print('state = {}'.format(state))
		#print('state.conj() = {}'.format(state.conj()))
		#print('state.shape  = {}'.format(state.shape))

		#print('trace  = {}'.format(trace))
		#print('measurement= {},   trace - measurement = {}'.format(measurement, trace - measurement))
		#print('diff   = {}'.format(diff))
		#print(' =============== ')
		#print('trAUU  = {}'.format(trAUU))
		#print('projU =\n{}'.format(projU))
		#print('diffU =\n{}'.format(diffU))

		return diffU


	def projection_diff(self):

		#state_diff = np.zeros(self.num_elements, dtype=complex)
		#stateU_Diff = np.zeros((self.num_elements, self.Nr), dtype=complex)
		stateU_Diff = np.zeros(self.stateU.shape, dtype=complex)
		
		for projector, measurement in zip(*[self.projector_list, self.measurement_list]):
			#state_diff += self.single_projection_diff(projector, measurement, self.momentum_state)

			#td1 = time.time()
			diffU = self.single_projU_diff(projector, measurement, self.momU, self.momUdag)			

			#print(' stateU_Diff = {}'.format(stateU_Diff))
			#print('       diffU = {}'.format(diffU))
			#print(' stateU_Diff.shape = {}'.format(stateU_Diff.shape))
			#print('       diffU.shape = {}'.format(diffU.shape))

			stateU_Diff += diffU
			#state_diff += diff
			#print(' state_diff  = \n {}'.format(state_diff))

			#td2 = time.time()
			#print('   diffU  --> time = {}'.format(td2-td1))

		#print(' state_diff =\n{}'.format(state_diff))
		#print(' stateU_Diff =\n{}'.format(stateU_Diff))

		#xx = stateU_Diff.T.conj()  @   stateU_Diff


		return stateU_Diff


	def step(self):

		#self.XkU_old                 = self.XkU
		self.previous_stateU         = self.stateU
		self.previous_momU           = self.momU
		self.Xk_old                  = self.Xk              #  added by Ming-Chien

		print(' --------  {}-th iteration \n'.format(self.iteration))
		tt1 = time.time()


		#self.previous_state          = self.state[:]
		#self.previous_momentum_state = self.momentum_state[:]
		
		stateU_Diff = self.projection_diff()
		tt2 = time.time()

		#state               = self.momentum_state - self.eta * state_diff
		stateU              = self.momU - self.eta * stateU_Diff

		#print(' {}-th ieration'.format(self.iteration))
		#print(' stateU_Diff = \n {}'.format(stateU_Diff))

		self.Unorm.append(np.linalg.norm(stateU))			#  to record the norm of state U

		#self.state          = clip_normalize(state, self.trace)
		#self.momentum_state = self.state + self.mu * (self.state - self.previous_state)

		self.stateU   = clip_normalize(stateU, self.trace)
		self.momU     = self.stateU + self.mu * (self.stateU - self.previous_stateU)
		self.momUdag  = self.momU.T.conj()

		tt3 = time.time()

		if self.iteration % 100 == 0:
			print(self.iteration)

		#print('  self.process_idx  = {}'.format(self.process_idx))
		#print('  self.iteration    = {}'.format(self.iteration))
		#print('  self.convergence_check_period  = {}'.format(self.convergence_check_period))

		#self.Xk = np.outer(self.state, self.state.T.conj())
		self.Xk = self.stateU @ self.stateU.T.conj()

		tt4 = time.time()

		self.convergence_check()
		tt5 = time.time()

		print('    projection_diff()  -->  time = {}'.format(tt2-tt1))
		print('    update variabels   -->  time = {}'.format(tt3-tt2))
		print('    calc  Xk           -->  time = {}'.format(tt4-tt3))
		print('    convergence_check  -->  time = {}'.format(tt5-tt4))
		print('    step (MiFGD)       -->  time = {}\n'.format(tt5 - tt1))

		self.step_Time[self.iteration] = tt5 - tt1


############################################################
## Utility functions
## XXX To modularize/package
############################################################

def density_matrix_norm(state):
	conj_state = state.conj()
	norm = np.sqrt(sum([v**2 for v in [np.linalg.norm(state * item)
									   for item in conj_state]]))
	return norm


def density_matrix_diff_norm(xstate, ystate):
	conj_xstate = xstate.conj()
	conj_ystate = ystate.conj()
	
	norm = np.sqrt(sum([v**2 for v in [np.linalg.norm(xstate * xitem - ystate * yitem)
									   for xitem, yitem in zip(*[conj_xstate, conj_ystate])]]))
	return norm



def clip_normalize(vector, threshold):
	norm = np.linalg.norm(vector)
	if norm  > threshold:
		vector = (threshold / norm) * vector
	return vector


## ------------------------------------------------ ##
##		MC personal writing for check				##
##		i.e. written by Ming-Chien					##
## ------------------------------------------------ ##

def Col_flip_sign(xCol):
    #ID_max_abs = np.argmax(np.abs( xCol))
    #sign = np.sign(xCol[ID_max_abs])

	#print(' ----------------------------------------------- ')

	if xCol.shape[0] <  xCol.shape[1]:
		print('  ***   ERROR: This is a row, not a column   ***')
		return 

	ID_max_abs = [np.argmax(np.abs( xCol[:, ii]))  for ii in range(xCol.shape[1])]
	sign = np.sign([xCol[ID_max_abs[ii], ii] for ii in range(xCol.shape[1])])
	
	#print('     xCol.shape = {}'.format(xCol.shape))
	#print('     sign.shape = {}'.format(sign.shape))
	#print('     sign       = {}'.format(sign))

	#print('     xCol = {}'.format(xCol[-5:]))
	#xCol = sign * xCol					#  may have diemsion problem
	xCol = np.multiply(sign, xCol)
	#print('     xCol = {}'.format(xCol[-5:]))
	#print('  ***** ')

	return xCol



def test_check_norm(state):
	"""	 similar to density_matrix_norm
	"""

	conj_state = state.conj()
	norm = np.sqrt(sum([v**2 for v in [np.linalg.norm(state * item)
									   for item in conj_state]]))

	xx = [item for item in conj_state]
	yy = [state * item for item in conj_state]
	zz = [np.linalg.norm(state * item) for item in conj_state]

	print(state)
	print(xx)
	print(yy)
	print(zz)

	return norm


def test_diff(xstate, ystate):
	"""	this is similar to def density_matrix_diff_norm(xstate, ystate):
	"""
	conj_xstate = xstate.conj()
	conj_ystate = ystate.conj()
	
	norm = np.sqrt(sum([v**2 for v in [np.linalg.norm(xstate * xitem - ystate * yitem)
									   for xitem, yitem in zip(*[conj_xstate, conj_ystate])]]))
	
	xx = [(xitem, yitem) for xitem, yitem in zip(*[conj_xstate, conj_ystate])]

	yy = [np.linalg.norm(xstate * xitem - ystate * yitem) 
       		for xitem, yitem in zip(*[conj_xstate, conj_ystate])]

	print(xx)
	print(yy)
	print(xstate)
	print(ystate)

	return norm
