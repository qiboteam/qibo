import inspect

import numpy as np
import sympy as sp

from qibo.config import raise_error


class Parameter:
    """Object which allows for variational gate parameters. Several trainable parameter
    and possibly features are linked through a lambda function which returns the
    final gate parameter. All possible analytical derivatives of the lambda function are
    calculated at the object initialisation using Sympy.

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
        self.nparams = len(trainable)

        if isinstance(feature, list):
            self.nfeat = len(feature)
        else:
            self.nfeat = 0

        # lambda function
        self.lambdaf = func
        self._check_inputs(func)

        self.derivatives = self._calculate_derivatives()

    def __call__(self):
        """Update values with trainable parameter and calculate current gate parameter"""
        return self._apply_func(self.lambdaf)

    def _check_inputs(self, func):
        """Verifies that the inputs are correct"""
        parameters = inspect.signature(func).parameters

        if (self.nfeat + self.nparams) != len(parameters):
            raise_error(
                ValueError,
                f"The lambda function has {len(parameters)} parameters, the input has {self.nfeat+self.nparams}.",
            )

        iterator = iter(parameters.items())

        for i in range(self.nfeat):
            x = next(iterator)
            if x[0][0] != "x":
                raise_error(
                    ValueError,
                    f"Parameter #{i} in the lambda function should be a feature starting with `x`",
                )

        for i in range(self.nparams):
            x = next(iterator)
            if x[0][:2] != "th":
                raise_error(
                    ValueError,
                    f"Parameter #{self.nfeat+i} in the lambda function should be a trainable parameter starting with `th`",
                )

    def _apply_func(self, function, fixed_params=None):
        """Applies lambda function and returns final gate parameter"""
        params = []
        if self._feature is not None:
            if isinstance(self._feature, list):
                params.extend(self._feature)
            else:
                params.append(self._feature)
        if fixed_params:
            params.extend(fixed_params)
        else:
            params.extend(self._trainable)
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
        if not isinstance(trainable, (list, np.ndarray)):
            raise_error(
                ValueError, "Trainable parameters must be given as list or numpy array"
            )

        if self.nparams != len(trainable):
            raise_error(
                ValueError,
                f"{len(trainable)} trainable parameters given, need {self.nparams}",
            )

        if not isinstance(feature, (list, np.ndarray)) and self._feature != feature:
            raise_error(ValueError, "Features must be given as list or numpy array")

        if self._feature is not None and self.nfeat != len(feature):
            raise_error(ValueError, f"{len(feature)} features given, need {self.nfeat}")

        if trainable is not None:
            self._trainable = trainable
        if feature and self._feature:
            self._feature = feature

    def get_indices(self, start_index):
        """Return list of respective indices of trainable parameters within
        a larger trainable parameter list"""
        return [start_index + i for i in range(self.nparams)]

    def get_fixed_part(self, trainable_idx):
        """Retrieve parameter constant unaffected by a specific trainable parameter"""
        params = self._trainable.copy()
        params[trainable_idx] = 0.0
        return self._apply_func(self.lambdaf, fixed_params=params)

    def get_partial_derivative(self, trainable_idx):
        """Get derivative w.r.t a trainable parameter"""
        deriv = self.derivatives[trainable_idx]
        return self._apply_func(deriv)
