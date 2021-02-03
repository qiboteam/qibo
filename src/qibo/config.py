"""
Define the default circuit, constants and types.
"""
import os
import blessings
import logging
import warnings

# Logging level from 0 (all) to 3 (errors)
LOG_LEVEL = 3

# Choose the least significant qubit
LEAST_SIGNIFICANT_QUBIT = 0

if LEAST_SIGNIFICANT_QUBIT != 0: # pragma: no cover
    # case not tested because least significant qubit is preset to 0
    raise_error(NotImplementedError, "The least significant qubit should be 0.")

# characters used in einsum strings
EINSUM_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Entanglement entropy eigenvalue cut-off
# Eigenvalues smaller than this cut-off are ignored in entropy calculation
EIGVAL_CUTOFF = 1e-14

# Flag for raising warning in ``set_precision`` and ``set_backend``
ALLOW_SWITCHERS = True


def raise_error(exception, message=None, args=None):
    """Raise exception with logging error.

    Args:
        exception (Exception): python exception.
        message (str): the error message.
    """
    log.error(message)
    if args:
        raise exception(message, args)
    else:
        raise exception(message)


# Set the number of threads from the environment variable
OMP_NUM_THREADS = None
if "OMP_NUM_THREADS" not in os.environ:
    import psutil
    # using physical cores by default
    cores = psutil.cpu_count(logical=False)
    OMP_NUM_THREADS = cores
else: # pragma: no cover
    OMP_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS"))

def get_threads():
    """Returns number of threads."""
    return OMP_NUM_THREADS

def set_threads(num_threads):
    """Set number of OpenMP threads.

    Args:
        num_threads (int): number of threads.
    """
    if not isinstance(num_threads, int): # pragma: no cover
        raise_error(RuntimeError, "Number of threads must be integer.")
    if num_threads < 1: # pragma: no cover
        raise_error(RuntimeError, "Number of threads must be positive.")
    global OMP_NUM_THREADS
    OMP_NUM_THREADS = num_threads


# Configuration for logging mechanism
t = blessings.Terminal()


class CustomColorHandler(logging.StreamHandler):
    """Custom color handler for logging algorithm."""

    colors = {
        logging.DEBUG: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold},
        logging.INFO: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold_green},
        logging.WARNING: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold_yellow},
        logging.ERROR: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold_red, '%(message)s': t.bold},
        logging.CRITICAL: {'[Qibo|%(levelname)s|%(asctime)s]:': t.bold_white_on_red, '%(message)s': t.bold},
    }

    def format(self, record):
        """Format the record with specific color."""
        levelcolors = self.colors[record.levelno]
        fmt = '[Qibo|%(levelname)s|%(asctime)s]: %(message)s'
        for s, subs in levelcolors.items():
            fmt = fmt.replace(s, subs(s))
        return logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S').format(record)


# allocate logger object
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(CustomColorHandler())


class StaticHardwareConfig:
    """Hardware backend parameters."""

    num_qubits = 2
    sampling_rate = 2.3e9
    nchannels = 4
    sample_size = 32000
    readout_pulse_type = "IQ"
    readout_pulse_duration = 5e-6
    readout_pulse_amplitude = 0.75
    lo_frequency = 4.51e9
    readout_nyquist_zone = 4
    ADC_sampling_rate = 2e9
    qubit_static_parameters = [
        {
            "id": 0,
            "channel": [2, None, [0, 1]], # XY control, Z line, readout
            "frequency_range": [2.6e9, 2.61e9],
            "resonator_frequency": 4.5241e9,
            "neighbours": [2]
        }, {
            "id": 1,
            "channel": [3, None, [0, 1]],
            "frequency_range": [3.14e9, 3.15e9],
            "resonator_frequency": 4.5241e9,
            "neighbours": [1]
        }
    ]
    dac_mode_for_nyquist = ["NRZ", "MIX", "MIX", "NRZ"] # fifth onwards not calibrated yet
    pulse_file = 'C:/fpga_python/fpga/tmp/wave_ch1.csv'


HW_PARAMS = StaticHardwareConfig()
