import numpy as np
import sympy as sp

from qibo.config import raise_error


class Parameter:
    """Object which allows for variational gate parameters. Several trainable parameter
    and possibly features are linked through a lambda function which returns the
    final gate parameter. All possible analytical derivatives of the lambda function are
    calculated at the object initialisation using Sympy.

    Example::

        from qibo.parameter import Parameter
        param = Parameter(
                lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
                [1.5, 2.0, 3.0],
                feature=[7.0],
            )

        partial_derivative = param.get_partial_derivative(3)

        param.update_parameters(trainable=[15.0, 10.0, 7.0], feature=[5.0])
        gate_value = param()


    Args:
        func (function): lambda function which builds the gate parameter. If both features and trainable parameters
        compose the function, it must be passed by first providing the features and then the parameters, as
        described in the code example above.
        trainable (list or np.ndarray): array with initial trainable parameters theta
        feature (list or np.ndarray): array containing possible input features x
    """

    def __init__(self, func, trainable, feature=None):
        self._trainable = trainable
        self._feature = feature

        # lambda function
        self.lambdaf = func

        self.derivatives = self._calculate_derivatives()

    def __call__(self):
        """Update values with trainable parameter and calculate current gate parameter"""
        return self._apply_func(self.lambdaf)

    @property
    def nparams(self):
        return len(self._trainable)

    @property
    def nfeat(self):
        return len(self._feature) if isinstance(self._feature, list) else 0

    def _apply_func(self, function, fixed_params=None):
        """Applies lambda function and returns final gate parameter"""
        params = []
        if self._feature is not None:
            params.extend(self._feature)
        if fixed_params:
            params.extend(fixed_params)
        else:
            params.extend(self._trainable)

        # run function
        return float(function(*params))

    def _calculate_derivatives(self):
        """Calculates derivatives w.r.t to all trainable parameters"""
        vars = []
        for i in range(self.nfeat):
            vars.append(sp.Symbol(f"x{i}"))
        for i in range(self.nparams):
            vars.append(sp.Symbol(f"th{i}"))

        expr = sp.sympify(self.lambdaf(*vars))

        derivatives = []
        for i in range(len(vars)):
            derivative_expr = sp.diff(expr, vars[i])
            derivatives.append(sp.lambdify(vars, derivative_expr))

        return derivatives

    def update_parameters(self, trainable=None, feature=None):
        """Update gate trainable parameter and feature values"""
        if trainable is not None:
            self._trainable = trainable

        if feature is not None and self._feature is not None:
            self._feature = feature

    def get_indices(self, start_index):
        """Return list of respective indices of trainable parameters within
        a larger trainable parameter list"""
        return (np.arange(self.nparams) + start_index).tolist()

    def get_fixed_part(self, trainable_idx):
        """Retrieve constant term of lambda function with regard to a specific trainable parameter"""
        params = self._trainable.copy()
        params[trainable_idx] = 0.0
        return self._apply_func(self.lambdaf, fixed_params=params)

    def get_partial_derivative(self, trainable_idx):
        """Get derivative w.r.t a trainable parameter"""
        deriv = self.derivatives[trainable_idx]
        return self._apply_func(deriv)
