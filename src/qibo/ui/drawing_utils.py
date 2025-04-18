from qibo.gates.abstract import Gate

QIBO_DEFAULT_COLOR = "#C194D8"
QIBO_COMPLEMENTARY_COLOR = "#DE8004"


class FusedStartGateBarrier(Gate):
    """
    :class:`qibo.ui.drawer_utils.FusedStartGateBarrier` gives room to fused group of gates.
    Inherit from ``qibo.gates.abstract.Gate``. A special gate barrier gate to pin the starting point of fused gates.
    """

    def __init__(self, q_ctrl, q_trgt, nfused, equal_qbits=False):

        super().__init__()
        self.name = (
            "FusedStartGateBarrier"
            + str(nfused)
            + ("" if not equal_qbits else "@EQUAL")
        )
        self.draw_label = ""
        self.control_qubits = (q_ctrl,)
        self.target_qubits = (q_trgt,) if q_ctrl != q_trgt else ()
        self.init_args = [q_trgt, q_ctrl] if q_ctrl != q_trgt else [q_ctrl]
        self.unitary = False
        self.is_controlled_by = False
        self.nfused = nfused


class FusedEndGateBarrier(Gate):
    """
    :class:`qibo.ui.drawer_utils.FusedEndGateBarrier` gives room to fused group of gates.
    Inherit from ``qibo.gates.abstract.Gate``. A special gate barrier gate to pin the ending point of fused gates.
    """

    def __init__(self, q_ctrl, q_trgt):

        super().__init__()
        self.name = "FusedEndGateBarrier"
        self.draw_label = ""
        self.control_qubits = (q_ctrl,)
        self.target_qubits = (q_trgt,) if q_ctrl != q_trgt else ()
        self.init_args = [q_trgt, q_ctrl] if q_ctrl != q_trgt else [q_ctrl]
        self.unitary = False
        self.is_controlled_by = False


def generate_bitstring_combinations(n):
    """Generate all bitstring combinations given bitstring length `n`."""
    bitstrings = []
    for i in range(2**n):
        bitstrings.append(format(i, f"0{n}b"))
    return bitstrings
