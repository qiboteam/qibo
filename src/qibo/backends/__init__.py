import os
from qibo import config
from qibo.config import raise_error, log

# versions requirements
TF_MIN_VERSION = '2.2.0'


class Backend:

    def __init__(self):

        self.available_backends = {}
        self.hardware_backends = {}
        active_backend = "numpy"

        # load profile from default file
        from pathlib import Path
        profile_path = Path(os.environ.get(
            'QIBO_PROFILE', Path(__file__).parent / "profiles.yml"))
        try:
            with open(profile_path) as f:
                import yaml
                profile = yaml.safe_load(f)
        except FileNotFoundError:  # pragma: no cover
            raise_error(FileNotFoundError,
                        f"Profile file {profile_path} not found.")

        # check if numpy is installed
        if self.check_availability("numpy"):
            from qibo.backends.numpy import NumpyBackend
            self.available_backends["numpy"] = NumpyBackend
        else:  # pragma: no cover
            raise_error(ModuleNotFoundError, "Numpy is not installed. "
                                             "Please install it using "
                                             "`pip install numpy`.")

        for backend in profile.get('backends'):
            name = backend.get('name')
            if self.check_availability(name):
                if name == 'tensorflow':
                    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(
                        config.TF_LOG_LEVEL)
                    import tensorflow as tf  # pylint: disable=E0401
                    if tf.__version__ < TF_MIN_VERSION:  # pragma: no cover
                        raise_error(
                            RuntimeError, f"TensorFlow version not supported, minimum is {TF_MIN_VERSION}.")
                import importlib
                custom_backend = getattr(importlib.import_module(
                    backend.get('from')), backend.get('class'))
                self.available_backends[name] = custom_backend
                if backend.get('is_hardware', False):
                    self.hardware_backends['qiboicarusq'] = custom_backend
                if profile.get('default') == name:
                    active_backend = name

        self.constructed_backends = {}
        self._active_backend = None
        self.qnp = self.construct_backend("numpy")
        # Create the default active backend
        if "QIBO_BACKEND" in os.environ:  # pragma: no cover
            self.active_backend = os.environ.get("QIBO_BACKEND")
        self.active_backend = active_backend

        # raise performance warning if qibojit and qibotf are not available
        self.show_config()
        if active_backend == "numpy":  # pragma: no cover
            log.warning("numpy backend uses `np.einsum` and supports CPU only. "
                        "Consider installing the qibojit or qibotf backends for "
                        "increased performance and to enable GPU acceleration.")
        elif active_backend == "tensorflow":  # pragma: no cover
            # case not tested because CI has tf installed
            log.warning("qibotf library was not found. `tf.einsum` will be "
                        "used to apply gates. In order to install Qibo's "
                        "high performance custom operators for TensorFlow "
                        "please use `pip install qibotf`. Alternatively, "
                        "consider installing the qibojit backend.")

    @property
    def active_backend(self):
        return self._active_backend

    @active_backend.setter
    def active_backend(self, name):
        self._active_backend = self.construct_backend(name)

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
            if name not in self.available_backends:
                available = [" - {}: {}".format(n, b.description)
                             for n, b in self.available_backends.items()]
                available = "\n".join(available)
                raise_error(ValueError, "Unknown backend {}. Please select one of "
                                        "the available backends:\n{}."
                                        "".format(name, available))
            new_backend = self.available_backends.get(name)()
            if self.active_backend is not None:
                new_backend.set_precision(self.active_backend.precision)
            self.constructed_backends[name] = new_backend
        return self.constructed_backends.get(name)

    def __getattr__(self, x):
        return getattr(self.active_backend, x)

    def __str__(self):
        return self.active_backend.name

    def __repr__(self):
        return str(self)

    def show_config(self):
        log.info(
            f"Using {self.active_backend.name} backend on {self.active_backend.default_device}")

    @staticmethod
    def check_availability(module_name):
        """Check if module is installed.

        Args:
            module_name (str): module name.

        Returns:
            True if the module is installed, False otherwise.
        """
        from pkgutil import iter_modules
        return module_name in (name for _, name, _ in iter_modules())


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
    K.active_backend = backend
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
