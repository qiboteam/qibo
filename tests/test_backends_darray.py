import dask.array as da
import pytest
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

import qibo.backends


@pytest.fixture
def da_backend():
    return qibo.backends.construct_backend("darray")


def test_darray(da_backend):
    n = 4
    state = da_backend.execute_circuit(qibo.models.QFT(n)).state()
    assert state.compute().size == 2**n
