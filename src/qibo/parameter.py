import numpy as np
import sympy as sp

from qibo.config import raise_error


class Parameter:
    """Object which allows for variational gate parameters. Several trainable parameters
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

    def __init__(self, func, features=None, trainable=None, nofeatures=False):
        self._trainable = trainable
        self._features = features
        self._nofeatures = nofeatures

        # lambda function
        self.lambdaf = func

        self.derivatives = self._calculate_derivatives()

    def __call__(self):
        """Update values with trainable parameter and calculate current gate parameter"""
        return self._apply_func(self.lambdaf)

    @property
    def nparams(self):
        """Returns the number of trainable parameters"""
        try:
            return len(self._trainable)
        except TypeError:
            return 0

    @property
    def nfeat(self):
        """Returns the number of features"""
        return len(self._features) if isinstance(self._features, list) else 0

    def _apply_func(self, function, fixed_params=None):
        """Applies lambda function and returns final gate parameter"""
        params = []
        if self._features is not None:
            params.extend(self._features)
        if fixed_params is not None:
            params.extend(fixed_params)
        else:
            params.extend(self._trainable)

        # run function
        return float(function(*params))

    def _calculate_derivatives(self):
        """Calculates derivatives w.r.t to all trainable parameters"""
        vars = []
        for i in range(self.lambdaf.__code__.co_argcount):
            vars.append(sp.Symbol(f"p{i}"))

        expr = sp.sympify(self.lambdaf(*vars))

        derivatives = []
        for i in range(len(vars)):
            derivative_expr = sp.diff(expr, vars[i])
            derivatives.append(sp.lambdify(vars, derivative_expr))

        return derivatives

    def gettrainable(self):
        return self._trainable

    def settrainable(self, value):
        self._trainable = value

    def getfeatures(self):
        return self._features

    def setfeatures(self, value):
        self._features = value if not self._nofeatures else None

    trainable = property(
        gettrainable, settrainable, doc="I'm the trainable parameters property."
    )
    features = property(getfeatures, setfeatures, doc="I'm the features property.")

    def trainable_parameter_indices(self, start_index):
        """Return list of respective indices of trainable parameters within
        the larger trainable parameter list of a circuit for example"""
        return (np.arange(self.nparams) + start_index).tolist()

    def unaffected_by(self, trainable_idx):
        """Retrieve constant term of lambda function with regard to a specific trainable parameter"""
        params = self._trainable.copy()
        params[trainable_idx] = 0.0
        return self._apply_func(self.lambdaf, fixed_params=params)

    def partial_derivative(self, trainable_idx):
        """Get derivative w.r.t a trainable parameter"""
        deriv = self.derivatives[trainable_idx]
        return self._apply_func(deriv)
