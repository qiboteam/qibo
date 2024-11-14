import numpy as np

from qibo import gates
from qibo.config import raise_error
from qibo.models.circuit import Circuit


def convert_bit_to_energy(hamiltonian, bitstring):
    """
    Given a binary string and a hamiltonian, we compute the corresponding energy.
    make sure the bitstring is of the right length
    """
    n = len(bitstring)
    circuit = Circuit(n)
    active_bit = [i for i in range(n) if bitstring[i] == "1"]
    for i in active_bit:
        circuit.add(gates.X(i))
    result = circuit()  # this is an execution result, a quantum state
    return hamiltonian.expectation(result.state())


def convert_state_to_count(state):
    """
    This is a function that convert a quantum state to a dictionary keeping track of
    energy and its frequency.
    """
    return np.abs(state) ** 2


def compute_cvar(probabilities, values, alpha, threshold=0.001):
    """
    Auxilliary method to computes CVaR for given probabilities, values, and confidence level.

    Args:
        probabilities (list): list/array of probabilities
        values (list): list/array of corresponding values
        alpha (float): confidence level
        threshold (float): a small positive number to avoid division by zero.

    Returns:
        CVaR
    """
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]
    vals = np.array(values)[sorted_indices]
    cum_probs = np.cumsum(probs)
    exceed_index = np.searchsorted(cum_probs, alpha, side="right")
    cvar = np.sum(probs[:exceed_index] * vals[:exceed_index]) / max(
        cum_probs[exceed_index - 1], threshold
    )  # avodiing division by 0
    return cvar


def cvar(hamiltonian, state, alpha=0.1):
    """
    Given the hamiltonian and state, this function estimate the
    corresponding cvar function
    """
    counts = convert_state_to_count(state)
    probabilities = np.zeros(len(counts))
    values = np.zeros(len(counts))
    m = int(np.log2(state.size))
    for i, p in enumerate(counts):
        values[i] = convert_bit_to_energy(hamiltonian, bin(i)[2:].zfill(m))
        probabilities[i] = p
    cvar_ans = compute_cvar(probabilities, values, alpha)
    return cvar_ans


def gibbs(hamiltonian, state, eta=0.1):
    """
    Given the hamiltonian and the state, and optional eta value
    it estimate the gibbs function value.
    """
    counts = convert_state_to_count(state)
    avg = 0
    sum_count = 0
    m = int(np.log2(state.size))
    for bitstring, count in enumerate(counts):
        obj = convert_bit_to_energy(hamiltonian, bin(bitstring)[2:].zfill(m))
        avg += np.exp(-eta * obj)
        sum_count += count
    return -np.log(avg / sum_count)


def initialize(nqubits: int, basis=gates.Z, eigenstate="+"):
    """This function returns a circuit that prepeares all the
    qubits in a specific state.

    Args:
        - nqubits (int): Number of qubit in the circuit.
        - baisis (gates): Can be a qibo gate or a callable that accepts a qubit,
        the default value is `gates.Z`.
        - eigenstate (str): Specify which eigenstate of the operator defined in
        `basis` will be the qubits' state. The default value is "+". Regarding the eigenstates
        of `gates.Z`, the `+` eigenstate is mapped in the zero state and the `-` eigenstate in the one state.
    """
    circuit_basis = Circuit(nqubits)
    circuit_eigenstate = Circuit(nqubits)
    if eigenstate == "-":
        for i in range(nqubits):
            circuit_eigenstate.add(gates.X(i))
    elif eigenstate != "+":
        raise_error(NotImplementedError, f"Invalid eigenstate {eigenstate}")

    for i in range(nqubits):
        value = basis(i).basis_rotation()
        if value is not None:
            circuit_basis.add(value)
    circuit_basis = circuit_basis.invert()
    return circuit_eigenstate + circuit_basis


def calculate_fourier_coeffs_unfiltered(input_function, frequency_degree):
    """Calculates the Fourier spectrum for a periodic function or quantum circuit within the range of the specified frequency degree.

    This function blindly computes the coefficients without applying any filtering and serves as a helper for the main
    ``fourier_coefficients`` function.

    Args:
        input_function (callable): A function that takes a 1D array of scalar inputs.
        frequency_degree (int or tuple[int]): The maximum frequency degree for which the Fourier coefficients will be computed.
            For a degree :math:`d`, the coefficients from frequencies :math:`-d, -d+1,...0,..., d-1, d` will be computed.

    Returns:
        array[complex]: The Fourier coefficients of the input function up to the specified degree.
    """
    frequency_degree = np.array(frequency_degree)
    number_of_coefficients = 2 * frequency_degree + 1

    # Create ranges of indices for each dimension
    index_ranges = [np.arange(-d, d + 1) for d in frequency_degree]

    # Generate all combinations of indices
    def product(*args):
        """Returns the cartesian product of the input iterables"""
        pools = [tuple(pool) for pool in args]
        result = [()]
        for pool in pools:
            result = [x + (y,) for x in result for y in pool]
        yield from result

    indices = product(*index_ranges)

    function_discretized = np.zeros(shape=tuple(number_of_coefficients))
    spacing = (2 * np.pi) / number_of_coefficients

    # Evaluate the function at each sampling point
    for index in indices:
        sampling_point = spacing * np.array(index)
        function_discretized[index] = input_function(sampling_point)

    # Compute the Fourier coefficients using the Fast Fourier Transform (FFT)
    coefficients = np.fft.fftn(function_discretized) / function_discretized.size
    return coefficients


def fourier_coefficients(
    f, n_inputs, degree, lowpass_filter=True, filter_threshold=None
):
    """Calculates the Fourier coefficients of a multivariate function up to a specified degree.
    This function can also compute the Fourier series for the expectation value of a quantum circuit.

    Args:
        f (callable): The input function to compute the Fourier coefficients for.
        n_inputs (int): The number of inputs (dimensions) of the function.
        degree (int or tuple[int]): The maximum degree of the Fourier series expansion.
        lowpass_filter (bool, optional): Flag to indicate whether to apply a low-pass filter to the coefficients. Default is True.
        filter_threshold (int or tuple[int], optional): The filter threshold for each input dimension. Default is None.

    Returns:
        array[complex]: The Fourier coefficients of the input function up to the specified degree.
    """
    if isinstance(degree, int):
        degree = (degree,) * n_inputs
    elif len(degree) != n_inputs:
        raise ValueError("The number of provided degrees must match n_inputs.")
    if not lowpass_filter:
        return calculate_fourier_coeffs_unfiltered(f, degree)
    if filter_threshold is None:
        filter_threshold = tuple(2 * d for d in degree)
    elif isinstance(filter_threshold, int):
        filter_threshold = (filter_threshold,) * n_inputs
    elif len(filter_threshold) != n_inputs:
        raise ValueError(
            "The number of provided filter thresholds must match n_inputs."
        )

    # Calculate unfiltered Fourier coefficients
    unfiltered_coeffs = calculate_fourier_coeffs_unfiltered(f, filter_threshold)
    # Shift the unfiltered coefficients
    shifted_unfiltered_coeffs = np.fft.fftshift(unfiltered_coeffs)
    shifted_filtered_coeffs = shifted_unfiltered_coeffs.copy()

    # Iterate to remove excess coefficients
    for axis in reversed(range(n_inputs)):
        num_excess = filter_threshold[axis] - degree[axis]
        slice_object = slice(
            num_excess, shifted_filtered_coeffs.shape[axis] - num_excess
        )
        shifted_filtered_coeffs = np.take(
            shifted_filtered_coeffs,
            np.arange(shifted_filtered_coeffs.shape[axis])[slice_object],
            axis=axis,
        )

    # Shift the filtered coefficients back
    filtered_coeffs = np.fft.ifftshift(shifted_filtered_coeffs)
    return filtered_coeffs


def vqe_loss(params, circuit, hamiltonian):
    circuit.set_parameters(params)
    result = hamiltonian.backend.execute_circuit(circuit)
    final_state = result.state()
    return hamiltonian.expectation(final_state)
