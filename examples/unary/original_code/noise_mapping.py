from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, pauli_error, thermal_relaxation_error
from numpy import array

"""
This file creates noise models for different kinds of error. The noise models are imported from qiskit
"""

def noise_model_measure(error, measure=True, thermal=True):
    """
    Creates errors for readout
    :param error: Probability of occuring a readout error / 10. This factor is added to maintain comparability with the rest of the code
    :return: noise model for the error
    """
    noise_model = NoiseModel()
    measure_error = 10 * error
    measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
    noise_model.add_all_qubit_readout_error(measure_error)

    return noise_model

def noise_model_phase(error, measure=True, thermal=True):
    """
    Creates error for phase flip
    :param error: Probability of happening a phaseflip
    :param measure: True or False, whether we include readout errors with probability 10 * error
    :param thermal: True or False, whether we include thermal relaxation, see noise_model_thermal
    :return: noise model for the error
    """
    noise_model = NoiseModel()#basis_gates=['id', 'u2', 'u3', 'cx'])
    phase_error = error
    cz_error = 2 * error
    phase_flip = pauli_error([('Z', phase_error), ('I', 1 - phase_error)])
    cz_error = pauli_error([('Z', cz_error), ('I', 1 - cz_error)])
    cz_error = cz_error.tensor(cz_error)
    noise_model.add_all_qubit_quantum_error(cz_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(phase_flip, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if thermal:
        thermal = thermal_relaxation_error(1.5, 1.2, error)
        cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
        cthermal = cthermal.tensor(cthermal)
        noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
        noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"], warnings=False)

    return noise_model

def noise_model_bit(error, measure=True, thermal=True):
    """
    Creates error for bit flip
    :param error: Probability of happening a bitflip
    :param measure: True or False, whether we include readout errors with probability 10 * error
    :param thermal: True or False, whether we include thermal relaxation, see noise_model_thermal
    :return: noise model for the error
    """
    noise_model = NoiseModel()
    flip_error = error
    cnot_error = 2 * error
    bit_flip = pauli_error([('X', flip_error), ('I', 1 - flip_error)])
    cnot_flip = pauli_error([('X', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_flip.tensor(cnot_flip)
    noise_model.add_all_qubit_quantum_error(bit_flip, ["u1", "u2", "u3"])
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'], warnings=False)


    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if thermal:
        thermal = thermal_relaxation_error(1.5, 1.2, error)
        cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
        cthermal = cthermal.tensor(cthermal)
        noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
        noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"], warnings=False)

    return noise_model


def noise_model_bitphase(error, measure=True, thermal=True):
    """
    Creates error for bitphase flip
    :param error: Probability of happening a bitphase flip
    :param measure: True or False, whether we include readout errors with probability 10 * error
    :param thermal: True or False, whether we include thermal relaxation, see noise_model_thermal
    :return: noise model for the error
    """
    noise_model = NoiseModel()  # basis_gates=['id', 'u2', 'u3', 'cx'])
    cnot_error = 2 * error
    bit_flip = pauli_error([('X', error), ('I', 1 - error)])
    phase_flip = pauli_error([('Z', error), ('I', 1 - error)])
    bitphase_flip = bit_flip.compose(phase_flip)
    cnot_flip = pauli_error([('X', cnot_error), ('I', 1 - cnot_error)])
    cnot_phase = pauli_error([('Z', cnot_error), ('I', 1 - cnot_error)])
    cnot_error = cnot_flip.compose(cnot_phase)
    cnot_error = cnot_error.tensor(cnot_error)
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(bitphase_flip, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if thermal:
        thermal = thermal_relaxation_error(1.5, 1.2, error)
        cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
        cthermal = cthermal.tensor(cthermal)
        noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
        noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"], warnings=False)

    return noise_model

def noise_model_depolarizing(error, measure=True, thermal=True):
    """
    Creates error for depolarizing channel
    :param error: Probability of depolarizing channel
    :param measure: True or False, whether we include readout errors with probability 10 * error
    :param thermal: True or False, whether we include thermal relaxation, see noise_model_thermal
    :return: noise model for the error
    """
    noise_model = NoiseModel()
    depolarizing = depolarizing_error(error, 1)
    cdepol_error = depolarizing_error(2 * error, 2)
    noise_model.add_all_qubit_quantum_error(cdepol_error, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(depolarizing, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if thermal:
        thermal = thermal_relaxation_error(1.5, 1.2, error)
        cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
        cthermal = cthermal.tensor(cthermal)
        noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
        noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"], warnings=False)

    return noise_model

def noise_model_thermal(error, measure=False, thermal=False):
    """
    Creates error for thermal relaxation for T1, T2 = 1.5, 1.2
    :param error: time of gate. Normalized to be equal to the error in all other functions
    :param measure: True or False, whether we include readout errors with probability 10 * error
    :return: noise model for the error
    """
    noise_model = NoiseModel()
    thermal = thermal_relaxation_error(1.5, 1.2, error)
    cthermal = thermal_relaxation_error(1.5, 1.2, 2 * error)
    cthermal = cthermal.tensor(cthermal)
    noise_model.add_all_qubit_quantum_error(cthermal, ['cx'], warnings=False)
    noise_model.add_all_qubit_quantum_error(thermal, ["u1", "u2", "u3"])

    if measure:
        measure_error = 10 * error
        measure_error = array([[1 - measure_error, measure_error], [measure_error, 1 - measure_error]])
        noise_model.add_all_qubit_readout_error(measure_error)

    return noise_model

