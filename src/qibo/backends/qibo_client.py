import qibo_client

from qibo.backends.numpy import NumpyBackend


class QiboClientBackend(NumpyBackend):
    """Backend for the remote execution of Qibo circuits.

    Args:
        platform (str): The destination client.
        token (str): User authentication token.
        runcard (dict): A dictionary containing the settings for the execution:
        - device (str): One of the devices supported by the platform.
    """

    def __init__(self, platform, token, runcard=None):
        super().__init__()
        if not runcard:
            runcard = {"device": "sim"}
        self.device = runcard["device"]
        self.client = getattr(qibo_client, platform)(token)

    def execute_circuit(self, circuit, nshots=1000):
        return self.client.run_circuit(circuit, nshots=nshots, device=self.device)
