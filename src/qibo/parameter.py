import numpy as np
import sympy as sp

from qibo.config import raise_error


def calculate_derivatives(func):
    """Calculates derivatives w.r.t. to all parameters of a target function `func`."""
    vars = []
    for i in range(func.__code__.co_argcount):
        vars.append(sp.Symbol(f"p{i}"))

    expr = sp.sympify(func(*vars))

    derivatives = []
    for i in range(len(vars)):
        derivative_expr = sp.diff(expr, vars[i])
        derivatives.append(sp.lambdify(vars, derivative_expr))

    return derivatives


class Parameter:
    """Object which allows for variational gate parameters. Several trainable parameters
    and possibly features are linked through a lambda function which returns the
    final gate parameter. All possible analytical derivatives of the lambda function are
    calculated at the object initialisation using Sympy.

    Example::

        from qibo.parameter import Parameter
        param = Parameter(
                lambda x, th1, th2, th3: x**2 * th1 + th2 * th3**2,
                features=[7.0],
                trainable=[1.5, 2.0, 3.0],
            )

        partial_derivative = param.get_partial_derivative(3)

        param.update_parameters(trainable=[15.0, 10.0, 7.0], feature=[5.0])
        param_value = param()


    Args:
        func (function): lambda function which builds the gate parameter. If both features and trainable parameters
            compose the function, it must be passed by first providing the features and then the parameters, as
            described in the code example above.
        features (list or np.ndarray): array containing possible input features x.
        trainable (list or np.ndarray): array with initial trainable parameters theta.
    """

    def __init__(self, func, trainable=None, features=None):
        self.trainable = trainable if trainable is not None else []
        self.features = features if features is not None else []

        if self.nfeat + self.nparams != func.__code__.co_argcount:
            raise_error(
                TypeError,
                f"{self.nfeat + self.nparams} parameters are provided, but {func.__code__.co_argcount} are required, please initialize features and trainable according to the defined function.",
            )
        # lambda function
        self.lambdaf = func

        # calculate derivatives
        # maybe here use JAX ?
        self.derivatives = calculate_derivatives(func=self.lambdaf)

    def __call__(self, features=None, trainable=None):
        """Return parameter value with given features and/or trainable."""

        params = []

        if features is None:
            params.extend(self.features)
        else:
            if len(features) != self.nfeat:
                raise_error(
                    TypeError,
                    f"The number of features provided is not compatible with the problem's dimensionality, which is {self.nfeat}.",
                )
            else:
                params.extend(features)
        if trainable is None:
            params.extend(self.trainable)
        else:
            if len(trainable) != self.nparams:
                raise_error(
                    TypeError,
                    f"The number of trainable provided is different from the number of required parameters, which is {self.nparams}.",
                )
            else:
                params.extend(trainable)

        return self.lambdaf(*params)

    @property
    def nparams(self):
        """Returns the number of trainable parameters"""
        return len(self.trainable)

    @property
    def nfeat(self):
        """Returns the number of features"""
        return len(self.features)

    @property
    def ncomponents(self):
        """Return the number of elements which compose the Parameter"""
        return self.nparams + self.nfeat

    def trainable_parameter_indices(self, start_index):
        """Return list of respective indices of trainable parameters within
        the larger trainable parameter list of a circuit for example"""
        return (np.arange(self.nparams) + start_index).tolist()

    def unaffected_by(self, trainable_idx):
        """Retrieve constant term of lambda function with regard to a specific trainable parameter"""
        params = self.trainable.copy()
        params[trainable_idx] = 0.0
        return self(trainable=params)

    def partial_derivative(self, trainable_idx):
        """Get derivative w.r.t a trainable parameter"""
        deriv = self.derivatives[trainable_idx]

        params = []
        params.extend(self.features)
        params.extend(self.trainable)

        return deriv(*params)
