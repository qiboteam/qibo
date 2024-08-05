from qibo.gates.abstract import Gate


class FusedStartGateBarrier(Gate):
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
    def __init__(self, q_ctrl, q_trgt):

        super().__init__()
        self.name = "FusedEndGateBarrier"
        self.draw_label = ""
        self.control_qubits = (q_ctrl,)
        self.target_qubits = (q_trgt,) if q_ctrl != q_trgt else ()
        self.init_args = [q_trgt, q_ctrl] if q_ctrl != q_trgt else [q_ctrl]
        self.unitary = False
        self.is_controlled_by = False
