import os
from qibo import config
from qibo.config import raise_error, log


class Backend:

    def __init__(self):
        # load profile from default file
        from pathlib import Path
        profile_path = Path(os.environ.get(
            'QIBO_PROFILE', Path(__file__).parent / "profiles.yml"))
        try:
            with open(profile_path) as f:
                import yaml
                self.profile = yaml.safe_load(f)
        except FileNotFoundError:  # pragma: no cover
            raise_error(FileNotFoundError, f"Profile file {profile_path} not found.")

        # dictionary to cache if backends are available
        # used by ``self.check_availability``
        self._availability = {}

        # create numpy backend (is always available as numpy is a requirement)
        if self.check_availability("numpy", check_version=False):
            from qibo.backends.numpy import NumpyBackend
            self.qnp = NumpyBackend()
        else:  # pragma: no cover
            raise_error(ModuleNotFoundError, "Numpy is not installed. "
                                             "Please install it using "
                                             "`pip install numpy`.")

        # loading backend names and version
        self._backends_min_version = {backend.get("name"): backend.get("minimum_version") for backend in self.profile.get("backends")}

        # find the default backend name
        default_backend = os.environ.get('QIBO_BACKEND', self.profile.get('default'))

        # check if default backend is described in the profile file
        if default_backend != "numpy":

            if default_backend not in (backend.get("name") for backend in self.profile.get("backends")): # pragma: no cover
                raise_error(ModuleNotFoundError, f"Default backend {default_backend} not set in {profile_path}.")

            # change the default backend if it is not available
            if not self.check_availability(default_backend):  # pragma: no cover
                # set the default backend to the first available declared backend in the profile file
                # if none is available it falls back to numpy
                for backend in self.profile.get('backends'):
                    name = backend.get('name')
                    if self.check_availability(name):
                        default_backend = name
                        break
                    # make numpy default if no other backend is available
                    default_backend = "numpy"

        self.active_backend = None
        self.constructed_backends = {"numpy": self.qnp}
        self.hardware_backends = {}
        # set default backend as active
        self.active_backend = self.construct_backend(default_backend)

        # raise performance warning if qibojit and qibotf are not available
        self.show_config()
        if str(self) == "numpy":  # pragma: no cover
            log.warning("numpy backend uses `np.einsum` and supports CPU only. "
                        "Consider installing the qibojit or qibotf backends for "
                        "increased performance and to enable GPU acceleration.")
        elif str(self) == "tensorflow":  # pragma: no cover
            # case not tested because CI has tf installed
            log.warning("qibotf library was not found. `tf.einsum` will be "
                        "used to apply gates. In order to install Qibo's "
                        "high performance custom operators for TensorFlow "
                        "please use `pip install qibotf`. Alternatively, "
                        "consider installing the qibojit backend.")

    @staticmethod
    def _get_backend_class(backend):
        """Loads class associated with the given backend.

        Helper method for ``construct_backend``.

        Args:
            backend (dict): The backend dictionary as read from ``profiles.yml``.

        Returns:
            Class that is used to initialize the given backend.
        """
        import importlib
        backend_module = importlib.import_module(backend.get('from'))
        return getattr(backend_module, backend.get('class'))

    def construct_backend(self, name):
        """Constructs and returns a backend.

        If the backend already exists in previously constructed backends then
        the existing object is returned.

        Args:
            name (str): Name of the backend to construct.
                See ``available_backends`` for the list of supported names.

        Returns:
            Backend object.
        """
        if name not in self.constructed_backends:
            backend = None
            if self.check_availability(name):
                for backend in self.profile.get('backends'):
                    if backend.get('name') == name:
                        break
            if backend is not None and backend.get('name') == name:
                backend_instance = self._get_backend_class(backend)()
                if self.active_backend is not None:
                    backend_instance.set_precision(self.active_backend.precision)
                self.constructed_backends[name] = backend_instance
                if backend.get('is_hardware', False):  # pragma: no cover
                    self.hardware_backends[name] = backend

            else:
                available = []
                for backend in self.profile.get('backends'):
                    n = backend.get('name')
                    if self.check_availability(n):
                        d = self._get_backend_class(backend).description
                        available.append(f" - {n}: {d}")
                available.append(f" - numpy: {self.qnp.description}")
                available = "\n".join(available)
                raise_error(ValueError, "Unknown backend {}. Please select one "
                                        "of the available backends:\n{}"
                                        "".format(name, available))

        return self.constructed_backends.get(name)

    def __getattr__(self, x):
        return getattr(self.active_backend, x)

    def __str__(self):
        return self.active_backend.name

    def __repr__(self):
        return str(self)

    def show_config(self):
        log.info(f"Using {self} backend on {self.active_backend.default_device}")

    def check_availability(self, module_name, check_version=True):
        """Check if module is installed.

        Args:
            module_name (str): module name.
            check_version (str): check if module has the required minimum version.
                A `ModuleNotFoundError` is raised if version is lower than minimum.

        Returns:
            ``True`` if the module is installed, ``False`` otherwise.
        """
        if module_name not in self._availability:
            from pkgutil import iter_modules
            is_available = module_name in (name for _, name, _ in iter_modules())
            if is_available and check_version:
                from importlib_metadata import version
                from packaging.version import parse
                minimum_version = self._backends_min_version.get(module_name, None)
                if minimum_version is not None and (parse(version(module_name)) < parse(minimum_version)): # pragma: no cover
                    raise_error(ModuleNotFoundError, f"Please upgrade {module_name}. "
                                                     f"Minimum supported version {minimum_version}.")
            self._availability[module_name] = is_available
        return self._availability.get(module_name)


K = Backend()
numpy_matrices = K.qnp.matrices


def set_backend(backend="qibojit"):
    """Sets backend used for mathematical operations and applying gates.

    The following backends are available:
    'qibojit': Numba/cupy backend with custom operators for applying gates,
    'qibotf': Tensorflow backend with custom operators for applying gates,
    'tensorflow': Tensorflow backend that applies gates using ``tf.einsum``,
    'numpy': Numpy backend that applies gates using ``np.einsum``.

    Args:
        backend (str): A backend from the above options.
    """
    if not config.ALLOW_SWITCHERS and backend != K.name:
        log.warning("Backend should not be changed after allocating gates.")
    K.active_backend = K.construct_backend(backend)
    K.show_config()


def get_backend():
    """Get backend used to implement gates.

    Returns:
        A string with the backend name.
    """
    return K.name


def set_precision(dtype="double"):
    """Set precision for states and gates simulation.

    Args:
        dtype (str): possible options are 'single' for single precision
            (complex64) and 'double' for double precision (complex128).
    """
    if not config.ALLOW_SWITCHERS and dtype != K.precision:
        log.warning("Precision should not be changed after allocating gates.")
    for bk in K.constructed_backends.values():
        bk.set_precision(dtype)


def get_precision():
    """Get precision for states and gates simulation.

    Returns:
        A string with the precision name ('single', 'double').
    """
    return K.precision


def set_device(name):
    """Set default execution device.

    Args:
        name (str): Device name. Should follow the pattern
            '/{device type}:{device number}' where device type is one of
            CPU or GPU.
    """
    if not config.ALLOW_SWITCHERS and name != K.default_device:
        log.warning("Device should not be changed after allocating gates.")
    K.set_device(name)
    for bk in K.constructed_backends.values():
        if bk.name != "numpy" and bk != K.active_backend:
            bk.set_device(name)


def get_device():
    return K.default_device


def set_threads(nthreads):
    """Set number of CPU threads.

    Args:
        nthreads (int): number of threads.
    """
    K.set_threads(nthreads)


def get_threads():
    """Returns number of threads."""
    return K.nthreads
