"""Model for combining trainable parameters and possible features into circuit parameters."""


class Parameter:
    def __init__(self, func, variational_parameters, features=None):
        self._variational_parameters = variational_parameters
        self._featurep = features
        self.nparams = len(variational_parameters)
        self.lambdaf = func

    def _apply_func(self, fixed_params=None):
        params = []
        if self._featurep is not None:
            params.append(self._featurep)
        if fixed_params:
            params.extend(fixed_params)
        else:
            params.extend(self._variational_parameters)
        return self.lambdaf(*params)

    def _update_params(self, trainablep=None, feature=None):
        if trainablep:
            self._variational_parameters = trainablep
        if feature:
            self._featurep = feature

    def get_params(self, trainablep=None, feature=None):
        self._update_params(trainablep=trainablep, feature=feature)
        return self._apply_func()

    def get_indices(self, start_index):
        return [start_index + i for i in range(self.nparams)]

    def get_fixed_part(self, trainablep_idx):
        params = [0] * self.nparams
        params[trainablep_idx] = self._variational_parameters[trainablep_idx]
        return self._apply_func(fixed_params=params)

    def get_scaling_factor(self, trainablep_idx):
        params = [0] * self.nparams
        params[trainablep_idx] = 1.0
        return self._apply_func(fixed_params=params)
