import re
import cirq
import numpy as np
_QIBO_TO_CIRQ = {"CNOT": "CNOT", "RY": "Ry", "TOFFOLI": "TOFFOLI"}


def assert_gates_equivalent(qibo_gate, cirq_gate):
    """Asserts that qibo gate is equivalent to cirq gate.

    Checks that:
        * Gate type agrees.
        * Target and control qubits agree.
        * Parameter (if applicable) agrees.

    Cirq gate parameters are extracted by parsing the gate string.
    """
    pieces = [x for x in re.split("[()]", str(cirq_gate)) if x]
    if len(pieces) == 2:
        gatename, targets = pieces
        theta = None
    elif len(pieces) == 3:
        gatename, theta, targets = pieces
    else: # pragma: no cover
        # case not tested because it fails
        raise RuntimeError("Cirq gate parsing failed with {}.".format(pieces))

    qubits = list(int(x) for x in targets.replace(" ", "").split(","))
    targets = (qubits.pop(),)
    controls = set(qubits)

    assert _QIBO_TO_CIRQ[qibo_gate.__class__.__name__] == gatename
    assert qibo_gate.target_qubits == targets
    assert set(qibo_gate.control_qubits) == controls
    if theta is not None:
        if "π" in theta:
            theta = np.pi * float(theta.replace("π", ""))
        else: # pragma: no cover
            # case doesn't happen in tests (could remove)
            theta = float(theta)
        np.testing.assert_allclose(theta, qibo_gate.parameters)
