import pytest


BACKENDS = ["custom"]

PRECISIONS = ["single", "double"]

NSHOTS = [int(1e3), int(1e5)]

# Set these according to machine specs
# or use parser?
NQUBITS = [10, 15]
NQUBITS_DISTRIBUTED = [(n, None) for n in NQUBITS]
# list of (nqubits, accelerators)
NQUBITS_DISTRIBUTED.extend([(20, {"/GPU:0": 1, "/GPU:1": 1})])


def pytest_generate_tests(metafunc):
    if "accelerators" in metafunc.fixturenames:
        assert "nqubits" in metafunc.fixturenames
        metafunc.parametrize("nqubits,accelerators", NQUBITS_DISTRIBUTED)
    else:
        if "nqubits" in metafunc.fixturenames:
            metafunc.parametrize("nqubits", NQUBITS)

    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", BACKENDS)
    if "precision" in metafunc.fixturenames:
        metafunc.parametrize("precision", PRECISIONS)
    if "nshots" in metafunc.fixturenames:
        metafunc.parametrize("nshots", NSHOTS)
