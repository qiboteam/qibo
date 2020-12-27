class Backend:

    base_methods = {"assign", "set_gates", "dtypes",
                    "set_precision", "set_device"}

    def __init__(self):
        self.gates = None
        self.custom_gates = True
        self.einsum = None

        self.precision = 'double'
        self._dtypes = {'DTYPEINT': 'int64', 'DTYPE': 'float64',
                        'DTYPECPX': 'complex128'}

        self.cpu_devices = None
        self.gpu_devices = None
        self.default_device = None

    def assign(self, backend):
        """Assigns backend's methods."""
        for method in dir(backend):
            if method[:2] != "__" and method not in self.base_methods:
                setattr(self, method, getattr(backend, method))

    def set_gates(self, name):
        if name == 'custom':
            self.custom_gates = True
            self.einsum = None
        elif name == 'defaulteinsum':
            from qibo.tensorflow import einsum
            self.custom_gates = False
            self.einsum = einsum.DefaultEinsum()
        elif name == 'matmuleinsum':
            from qibo.tensorflow import einsum
            self.custom_gates = False
            self.einsum = einsum.MatmulEinsum()
        else:
            raise_error(RuntimeError, f"Gate backend '{name}' not supported.")
        self.gates = name

    def dtypes(self, name):
        if name in self._dtypes:
            dtype = self._dtypes.get(name)
        else:
            dtype = name
        return getattr(self.backend, dtype)

    def set_precision(self, dtype):
        from qibo.config import ALLOW_SWITCHERS, warnings
        if not ALLOW_SWITCHERS and dtype != self.precision:
            warnings.warn("Precision should not be changed after allocating gates.",
                          category=RuntimeWarning)
        if dtype == 'single':
            self._dtypes['DTYPE'] = 'float32'
            self._dtypes['DTYPECPX'] = 'complex64'
        elif dtype == 'double':
            self._dtypes['DTYPE'] = 'float64'
            self._dtypes['DTYPECPX'] = 'complex128'
        else:
            raise_error(RuntimeError, f'dtype {dtype} not supported.')
        self.precision = dtype

    def set_device(self, name):
        from qibo.config import ALLOW_SWITCHERS, warnings
        if not ALLOW_SWITCHERS and name != self.default_device: # pragma: no cover
            # no testing is implemented for warnings
            warnings.warn("Device should not be changed after allocating gates.",
                          category=RuntimeWarning)
        parts = name[1:].split(":")
        if name[0] != "/" or len(parts) < 2 or len(parts) > 3:
            raise_error(ValueError, "Device name should follow the pattern: "
                             "/{device type}:{device number}.")
        device_type, device_number = parts[-2], int(parts[-1])
        if device_type not in {"CPU", "GPU"}:
            raise_error(ValueError, f"Unknown device type {device_type}.")
        if device_number >= len(self._devices[device_type]):
            raise_error(ValueError, f"Device {name} does not exist.")
        self.default_device = name


backend = Backend()
