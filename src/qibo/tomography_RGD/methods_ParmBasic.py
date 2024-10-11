#
#   get basic parameter information for tomography 
#   


class BasicParameterInfo:
    def __init__(self, params_dict):

        #
        #	read in params_dict
        #
        Nr               = params_dict.get('Nr', 1)
        trace            = params_dict.get('trace', 1.0)
        target_state     = params_dict.get('target_state', None)
        target_DM		 = params_dict.get('target_DM')
    
        label_list       = params_dict.get('labels')
        measurement_list = params_dict.get('measurement_list') 
        projector_list   = params_dict.get('projector_list')

        num_iterations   = params_dict.get('num_iterations')
        convergence_check_period = params_dict.get('convergence_check_period', 10)

        relative_error_tolerance = params_dict.get('relative_error_tolerance', 0.001)
        seed                     = params_dict.get('seed', 0)

        # 
        # basic system information
        #
        n = len(label_list[0])
        d = 2 ** n
		
        self.n            = n   # number of qubits
        self.num_elements = d
        self.Nr           = Nr  # the rank of the target_density_matrix

        self.trace            = trace
        self.target_state     = target_state
        self.target_DM		  = target_DM			#  target state -->  density matrix

        self.num_labels       = len(label_list)
        self.measurement_list = measurement_list
        self.projector_list   = projector_list

        # 
        #	numerical book keeping
        #

        self.process_idx				= 0

        self.num_iterations				= num_iterations

        self.convergence_check_period   = convergence_check_period   # how often to check convergence
        self.relative_error_tolerance   = relative_error_tolerance   # user-decided relative error 
        self.seed                       = seed


        self.Err_relative_st       		= []

        self.Target_Err_st              = []	# 	Frobenius norm difference from State
        self.Target_Err_Xk				= [] 	#   Frobenius norm difference from matrix Xk

        self.target_error_list          = []
        self.target_relative_error_list = []

        self.fidelity_Xk_list			= []	#   Fidelity between (Xk, target)
        self.Err_relative_Xk            = []

        self.iteration                  = 0
        self.converged                  = False
        self.convergence_iteration      = 0

        self.fidelity_list              = []
