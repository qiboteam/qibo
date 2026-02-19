"""Module with functions that encode classical data into quantum circuits."""

import math
from inspect import signature
from typing import List, Optional, Union

import numpy as np
from scipy.special import binom

from qibo import gates
from qibo.backends import Backend, _check_backend
from qibo.config import raise_error
from qibo.models.circuit import Circuit
from qibo.models._encodings import (
    _add_dicke_unitary_gate,
    _add_wbd_gate,
    _angle_mod_two_pi,
    _binary_encoder_hopf,
    _binary_encoder_hyperspherical,
    _ehrlich_algorithm,
    _generate_rbs_angles,
    _generate_rbs_pairs,
    _get_gate,
    _get_phase_gate_correction,
    _non_trivial_layers,
    _parametrized_two_qubit_gate,
    _perm_column_ops,
    _perm_pair_flip_ops,
    _perm_row_ops,
    _sparse_encoder_farias,
    _sparse_encoder_li,
    _up_to_k_encoder_hyperspherical,
)


def binary_encoder(
    data,
    parametrization: str = "hyperspherical",
    nqubits: int = None,
    codewords=None,
    keep_antictrls: bool = False,
    backend=None,
    **kwargs,
):
    """Create circuit that encodes :math:`1`-dimensional data in all amplitudes
    of the computational basis.

    Given data vector :math:`\\mathbf{x} \\in \\mathbb{C}^{d}`, with :math:`d = 2^{n}`,
    this function generates a quantum circuit :math:`\\mathrm{Load}` that encodes
    :math:`\\mathbf{x}` in the amplitudes of an :math:`n`-qubit quantum state as

    .. math::
        \\mathrm{Load}(\\mathbf{x}) \\, \\ket{0}^{\\otimes \\, n} = \\sum_{j=0}^{d-1} \\,
            \\frac{x_{j}}{\\|\\mathbf{x}\\|_{F}} \\, \\ket{b_{j}} \\, ,

    where :math:`b_{j} \\in \\{0, \\, 1\\}^{\\otimes \\, n}` is the :math:`n`-bit representation
    of the integer :math:`j`, :math:`\\|\\cdot\\|_{F}` is the Frobenius norm.

    Resulting circuit parametrizes ``data`` in either ``hyperspherical`` or ``hopf`` coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.

    Args:
        data (ndarray): :math:`1`-dimensional array of length :math:`d = 2^{n}`
            to be loaded in the amplitudes of a :math:`n`-qubit quantum state.
        parametrization (str): choice of circuit parametrization. either ``hyperspherical``
            or ``hopf`` coordinates in the :math:`(2^{n} - 1)`-unit sphere.
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        codewords (int, optional): list of codewords. When parametrization is ``hyperspherical``,
            the list is used to encode the data in the given order. If ``None``,
            the codewords are set by the erhlich algorithm.
        keep_antictrls (bool, optional): If ``True`` and parametrization is ``hyperspherical``, we
            don't simplify the anti-controls when placing the RBS gates. For details, see [1].
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in binary encoding.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.

        2. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.

        3. H. S. Cohl, *Fourier, Gegenbauer and Jacobi expansions for a power-law fundamental
        solution of the polyharmonic equation and polyspherical addition theorems*, `Symmetry,
        Integrability and Geometry: Methods and Applications 10.3842/sigma.2013.042 (2013)
        <https://arxiv.org/abs/1209.6047>`_.
    """
    backend = _check_backend(backend)

    dims = len(data)
    if (dims & (dims - 1)) != 0 and parametrization == "hopf":
        raise_error(ValueError, "`data` size must be a power of 2.")

    if nqubits is None:
        nqubits = int(backend.ceil(backend.log2(dims)))

    complex_data = bool(
        "complex" in str(data.dtype)
    )  # backend-agnostic way of checking the dtype

    if parametrization == "hopf":
        return _binary_encoder_hopf(
            data, nqubits, complex_data=complex_data, backend=backend, **kwargs
        )

    return _binary_encoder_hyperspherical(
        data,
        nqubits,
        complex_data=complex_data,
        backend=backend,
        codewords=codewords,
        keep_antictrls=keep_antictrls,
        **kwargs,
    )


def comp_basis_encoder(
    basis_element: Union[int, str, list, tuple], nqubits: Optional[int] = None, **kwargs
):
    """Create circuit that performs encoding of bitstrings into computational basis states.

    Args:
        basis_element (int or str or list or tuple): bitstring to be encoded.
            If ``int``, ``nqubits`` must be specified.
            If ``str``, must be composed of only :math:`0`s and :math:`1`s.
            If ``list`` or ``tuple``, must be composed of :math:`0`s and
            :math:`1`s as ``int`` or ``str``.
        nqubits (int, optional): total number of qubits in the circuit.
            If ``basis_element`` is ``int``, ``nqubits`` must be specified.
            If ``nqubits`` is ``None``, ``nqubits`` defaults to length of ``basis_element``.
            Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
       :class:`qibo.models.circuit.Circuit`: Circuit encoding computational basis element.
    """
    if not isinstance(basis_element, (int, str, list, tuple)):
        raise_error(
            TypeError,
            "basis_element must be either type int or str or list or tuple, "
            + f"but it is type {type(basis_element)}.",
        )

    if isinstance(basis_element, (str, list, tuple)):
        if any(elem not in ["0", "1", 0, 1] for elem in basis_element):
            raise_error(ValueError, "all elements must be 0 or 1.")

    if nqubits is not None and not isinstance(nqubits, int):
        raise_error(
            TypeError, f"``nqubits`` must be type int, but it is type {type(nqubits)}."
        )

    if nqubits is None:
        if isinstance(basis_element, int):
            raise_error(
                ValueError,
                "``nqubits`` must be specified when ``basis_element`` is type int.",
            )
        else:
            nqubits = len(basis_element)

    if isinstance(basis_element, int):
        basis_element = f"{basis_element:0{nqubits}b}"

    if isinstance(basis_element, (str, tuple)):
        basis_element = list(basis_element)

    basis_element = list(map(int, basis_element))

    circuit = Circuit(nqubits, **kwargs)
    for qubit, elem in enumerate(basis_element):
        if elem == 1:
            circuit.add(gates.X(qubit))

    return circuit


def dicke_state(nqubits: int, weight: int, all_to_all: bool = False, **kwargs):
    """Create a circuit that prepares the Dicke state :math:`\\ket{D_{k}^{n}}`.

    The Dicke state :math:`\\ket{D_{k}^{n}}` is the equal superposition of all :math:`n`-qubit
    computational basis states with fixed-Hamming-weight :math:`k`.
    The circuit prepares the state deterministically with :math:`O(k \\, n)` gates and :math:`O(n)`
    depth, or :math:`O(k \\log\\frac{n}{k})` depth under the assumption of ``all-to-all``
    connectivity.

    Args:
        nqubits (int): number of qubits :math:`n`.
        weight (int): Hamming weight :math:`k` of the Dicke state.
        all_to_all (bool, optional): If ``False``, uses implementation from Ref. [1].
            If ``True``, uses shorter-depth implementation from Ref. [2].
            Defaults to ``False``.
        kwargs (dict, optional): additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit` : Circuit that prepares :math:`\\ket{D_{k}^{n}}`.

    References:
        1. Andreas Bärtschi and Stephan Eidenbenz, *Deterministic preparation of Dicke states*,
        `22nd International Symposium on Fundamentals of Computation Theory, FCT'19, 126-139  (2019)
        <https://doi.org/10.1007/978-3-030-25027-0_9>`_.

        2. Andreas Bärtschi and Stephan Eidenbenz, *Short-Depth Circuits for Dicke State Preparation*,
        `IEEE International Conference on Quantum Computing & Engineering (QCE), 87--96 (2022)
        <https://doi.org/10.1109/QCE53715.2022.00027>`_.
    """
    if weight < 0 or weight > nqubits:
        raise_error(
            ValueError, f"weight must be between 0 and {nqubits}, but got {weight}."
        )

    circuit = Circuit(nqubits, **kwargs)

    if weight == 0:
        return circuit

    if not all_to_all:
        # Start with |0⟩^(n-k) |1⟩^k
        circuit.add(gates.X(qubit) for qubit in range(nqubits - weight, nqubits))

        _add_dicke_unitary_gate(circuit, range(nqubits), weight)

        return circuit

    # We prepare disjoint sets of qubits
    disjoint_sets = [
        {
            "qubits": list(range(weight * it, weight * (it + 1))),
            "tier": weight,
            "children": [],
        }
        for it in range(nqubits // weight)
    ]
    nmodk = nqubits % weight
    if nmodk != 0:
        disjoint_sets.append(
            {
                "qubits": list(range(nqubits - nmodk, nqubits)),
                "tier": nmodk,
                "children": [],
            }
        )
    # reverse to have in ascending order of tier
    disjoint_sets = list(reversed(disjoint_sets))

    trees = disjoint_sets.copy()
    # combine lowest tier trees into one tree
    while len(trees) > 1:
        first_smallest = trees.pop(0)
        second_smallest = trees.pop(0)
        second_smallest["tier"] += first_smallest["tier"]
        second_smallest["children"].append(first_smallest)
        new = second_smallest
        # put new combined tree in list mantaining ordering
        trees.insert(sum(x["tier"] < new["tier"] for x in trees), new)

    root = trees[0]
    # We initialize |0>^(n-k)|1>^k  bitstring at root qubits
    circuit.add(gates.X(q) for q in root["qubits"][-weight:])

    # undo the union-by-tier algorithm:
    # split each root's (x) highest tier child y, and add WBD acting on both
    while len(trees) < len(disjoint_sets):
        cut_nodes = []
        for node in trees:
            if len(node["children"]) > 0:
                # find highest tier child
                y = max(node["children"], key=lambda x: x["tier"])
                # add WBD acting on both sets of qubits
                _add_wbd_gate(
                    circuit,
                    node["qubits"],
                    y["qubits"],
                    node["tier"],
                    y["tier"],
                    weight,
                )
                # cut tree splitting x and y
                node["tier"] -= y["tier"]
                node["children"].remove(y)
                cut_nodes.append(y)
        trees += cut_nodes

    for node in disjoint_sets:
        _add_dicke_unitary_gate(
            circuit, node["qubits"], min(weight, len(node["qubits"]))
        )

    return circuit


def entangling_layer(
    nqubits: int,
    architecture: str = "diagonal",
    entangling_gate: Union[str, gates.Gate] = "CNOT",
    closed_boundary: bool = False,
    **kwargs,
):
    """Create a layer of two-qubit entangling gates.

    If the chosen gate is a parametrized gate, all phases are set to :math:`0.0`.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        architecture (str, optional): Architecture of the entangling layer.
            In alphabetical order, options are ``"diagonal"``, ``"even_layer"``,
            ``"next_nearest"``, ``"odd_layer"``, ``"pyramid"``, ``"shifted"``,
            ``"v"``, and ``"x"``. The ``"x"`` architecture is only defined for an even number
            of qubits. Defaults to ``"diagonal"``.
        entangling_gate (str or :class:`qibo.gates.Gate`, optional): Two-qubit gate to be used
            in the entangling layer. If ``entangling_gate`` is a parametrized gate,
            all phases are initialized as :math:`0.0`. Defaults to  ``"CNOT"``.
        closed_boundary (bool, optional): If ``True`` and ``architecture not in
            ["pyramid", "v", "x"]``, adds a closed-boundary condition to the entangling layer.
            Defaults to ``False``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit containing layer of two-qubit gates.
    """

    if not isinstance(nqubits, int):
        raise_error(
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )

    if nqubits <= 0.0:
        raise_error(
            ValueError, f"nqubits must be a positive integer, but it is {nqubits}."
        )

    if not isinstance(architecture, str):
        raise_error(
            TypeError,
            f"``architecture`` must be type str, but it is type {type(architecture)}.",
        )

    if architecture not in [
        "diagonal",
        "even_layer",
        "next_nearest",
        "odd_layer",
        "pyramid",
        "shifted",
        "v",
        "x",
    ]:
        raise_error(
            NotImplementedError,
            f"``architecture`` {architecture} not found.",
        )

    if architecture == "x" and nqubits % 2 != 0.0:
        raise_error(
            ValueError, "``x`` architecture only defined for an even number of qubits."
        )

    if not isinstance(closed_boundary, bool):
        raise_error(
            TypeError,
            f"``closed_boundary`` must be type bool, but it is type {type(closed_boundary)}.",
        )

    gate = (
        getattr(gates, entangling_gate)
        if isinstance(entangling_gate, str)
        else entangling_gate
    )

    if gate.__name__ == "GeneralizedfSim":
        raise_error(
            NotImplementedError,
            "This function does not support the ``GeneralizedfSim`` gate.",
        )

    if architecture in ["next_nearest", "pyramid", "v", "x"]:
        circuit = _non_trivial_layers(
            nqubits,
            architecture=architecture,
            entangling_gate=entangling_gate,
            closed_boundary=closed_boundary,
            **kwargs,
        )
    else:
        # Finds the correct number of parameters to initialize the gate class.
        parameters = list(signature(gate).parameters)

        if "q2" in parameters:
            raise_error(
                NotImplementedError, "This function does not accept three-qubit gates."
            )

        # If gate is parametrized, sets all angles to 0.0
        parameters = (0.0,) * (len(parameters) - 3) if len(parameters) > 2 else None

        circuit = Circuit(nqubits, **kwargs)

        if architecture == "diagonal":
            qubits = range(nqubits - 1)
        elif architecture == "even_layer":
            qubits = range(0, nqubits - 1, 2)
        elif architecture == "odd_layer":
            qubits = range(1, nqubits - 1, 2)
        else:
            qubits = tuple(range(0, nqubits - 1, 2)) + tuple(range(1, nqubits - 1, 2))

        circuit.add(
            _parametrized_two_qubit_gate(gate, qubit, qubit + 1, parameters)
            for qubit in qubits
        )

        if closed_boundary:
            circuit.add(_parametrized_two_qubit_gate(gate, nqubits - 1, 0, parameters))

    return circuit


def ghz_state(nqubits: int, **kwargs):
    """Generate an :math:`n`-qubit Greenberger-Horne-Zeilinger (GHZ) state that takes the form

    .. math::
        \\ket{\\text{GHZ}} = \\frac{\\ket{0}^{\\otimes n} + \\ket{1}^{\\otimes n}}{\\sqrt{2}}

    where :math:`n` is the number of qubits.

    Args:
        nqubits (int): number of qubits :math:`n >= 2`.
        kwargs (dict, optional): additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that prepares the GHZ state.
    """
    if nqubits < 2:
        raise_error(
            ValueError,
            f"nqubits given as {nqubits}. nqubits needs to be >= 2.",
        )

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(qubit, qubit + 1) for qubit in range(nqubits - 1))

    return circuit


def graph_state(matrix, backend=None, **kwargs):
    """Create circuit encoding an undirected graph state given its adjacency matrix.

    Given a graph :math:`G = (V, E)` with :math:`V` being the set of vertices and :math:`E`
    being the set of edges, an $n$-qubit graph state is defined as

    .. math::
        \\ket{G} = \\prod_{(j, k) \\in E} \\, CZ_{j, k} \\, \\ket{+}^{\\otimes V} \\, ,

    where :math:`CZ_{a,b}` is the :class:`qibo.gates.CZ` gate acting on qubits :math:`j`
    and :math:`k`, and :math:`\\ket{+} = H \\, \\ket{0}$, with :class:`qibo.gates.H`
    being the Hadamard gate.

    Args:
        matrix (ndarray or list): Adjacency matrix of the graph.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit of the graph state with the given
        adjacency matrix.
    """
    backend = _check_backend(backend)

    if isinstance(matrix, list):
        matrix = backend.cast(matrix, dtype=int)

    if not backend.allclose(matrix, matrix.T):
        raise_error(
            ValueError,
            f"``matrix`` is not symmetric, not representing an undirected graph",
        )

    nqubits = len(matrix)

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.H(qubit) for qubit in range(nqubits))

    # since the matrix is symmetric, we only need the upper triangular part
    rows, columns = backend.nonzero(backend.triu(matrix))
    circuit.add(gates.CZ(int(ind_r), int(ind_c)) for ind_r, ind_c in zip(rows, columns))

    return circuit


def hamming_weight_encoder(
    data,
    nqubits: int,
    weight: int,
    full_hwp: bool = False,
    optimize_controls: bool = True,
    phase_correction: bool = True,
    initial_string=None,
    backend=None,
    **kwargs,
):
    """Create circuit that encodes ``data`` in the Hamming-weight-:math:`k` basis of ``nqubits``.

    Let :math:`\\mathbf{x}` be a :math:`1`-dimensional array of size :math:`d = \\binom{n}{k}` and
    :math:`B_{k} \\equiv \\{ \\ket{b_{j}} : b_{j} \\in \\{0, 1\\}^{\\otimes n} \\,\\, \\text{and}
    \\,\\, |b_{j}| = k \\}` be a set of :math:`d` computational basis states of :math:`n` qubits
    that are represented by bitstrings of Hamming weight :math:`k`. Then, an amplitude encoder
    in the basis :math:`B_{k}` is an :math:`n`-qubit parameterized quantum circuit
    :math:`\\operatorname{Load}_{B_{k}}` such that

    .. math::
        \\operatorname{Load}(\\mathbf{x}) \\, \\ket{0}^{\\otimes n} = \\frac{1}{\\|\\mathbf{x}\\|}
            \\, \\sum_{j = 1}^{d} \\, x_{j} \\, \\ket{b_{j}}

    Args:
        data (ndarray): :math:`1`-dimensional array of data to be loaded.
        nqubits (int): number of qubits.
        weight (int): Hamming weight that defines the subspace in which ``data`` will be encoded.
        full_hwp (bool, optional): if ``False``, includes Pauli-:math:`X` gates that prepare the
            first bitstring of Hamming weight ``k = weight``. If ``True``, circuit is full
            Hamming weight preserving. Defaults to ```False``.
        optimize_controls (bool, optional): if ``True``, removes unnecessary controlled operations.
            Defaults to ``True``.
        phase_correction (bool, optional): To be used when ``data`` is complex-valued.
            If ``True``, adds a controlled-$\\mathrm{RZ}$ gate to the end of the circuit,
            adding a final phase correction. If ``False``, gate is not added. Defaults to ``True``.
        initial_string (ndarray, optional): Array containing the desired initial bitstring of
            Hamming ``weight`` $k$. If ``None``, defaults to $\\ket{1^{k}0^{n-k}}$.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in
        Hamming-weight-:math:`k` representation.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.
    """
    backend = _check_backend(backend)

    complex_data = bool("complex" in str(data.dtype))

    if initial_string is None:
        initial_string = np.array([1] * weight + [0] * (nqubits - weight))
    bitstrings, targets_and_controls = _ehrlich_algorithm(initial_string)

    # sort data such that the encoding is performed in lexicographical order
    lex_order = [int(string, 2) for string in bitstrings]
    lex_order_sorted = np.sort(np.copy(lex_order))
    lex_order = np.array([np.where(lex_order_sorted == num)[0][0] for num in lex_order])
    data = data[lex_order]
    del lex_order, lex_order_sorted

    # Calculate all gate phases necessary to encode the amplitudes.
    _data = backend.abs(data) if complex_data else data
    thetas = _generate_rbs_angles(_data, architecture="diagonal", backend=backend)
    phis = backend.zeros(len(thetas) + 1, dtype=float)
    if complex_data:
        phis[0] = _angle_mod_two_pi(-backend.angle(data[0]))
        for k in range(1, len(phis)):
            phis[k] = _angle_mod_two_pi(-backend.angle(data[k]) + backend.sum(phis[:k]))

    last_qubit = nqubits - 1

    circuit = Circuit(nqubits, **kwargs)
    if not full_hwp:
        circuit.add(
            gates.X(qubit) for qubit in range(last_qubit, last_qubit - weight, -1)
        )

    if optimize_controls:
        indices = [
            int(binom(nqubits - j, weight - j)) - 1 for j in range(weight - 1, 0, -1)
        ]

    for k, ((targets, controls), theta, phi) in enumerate(
        zip(targets_and_controls, thetas, phis)
    ):
        targets = list(last_qubit - np.asarray(targets))
        controls = list(last_qubit - np.asarray(controls))
        controls.sort()

        if optimize_controls:
            controls = list(np.asarray(controls)[k >= np.asarray(indices)])

        gate = _get_gate(
            [targets[0]],
            [targets[1]],
            controls,
            theta,
            phi,
            complex_data,
        )
        circuit.add(gate)

    if complex_data and phase_correction:
        circuit.add(_get_phase_gate_correction(bitstrings[-1], phis[-1]))

    return circuit


def permutation_synthesis(
    sigma: Union[List[int], tuple[int, ...]], m: int = 2, backend=None, **kwargs
):
    """Return circuit that implements a given permutation.

    Given permutation ``sigma`` on :math:`\\{0, \\, 1, \\, \\dots, \\, d-1\\}`
    and a power‑of‑two budget ``m``, this function factors ``sigma``
    into the fewest layers :math:`\\sigma_{1}, \\, \\sigma_{2}, \\, \\cdots, \\, \\sigma_{t}`
    such that:
        - each layer has at most :math:`m` disjoint transpositions;
        - each layer moves a power‑of‑two number of indices.

    The function returns a circuit synthesis of ``sigma``.

    Args:
        sigma (list or tuple): permutation description on :math:`\\{0, \\, 1, \\, \\dots, \\, d-1\\}`.
        m (int): power‑of‑two budget. Defauls to :math:`2`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that implements the permutation ``sigma``.

    References:
        1. L. Li, and J. Luo,
        *Nearly Optimal Circuit Size for Sparse Quantum State Preparation*
        `arXiv:2406.16142 (2024) <https://doi.org/10.48550/arXiv.2406.16142>`_.
    """
    backend = _check_backend(backend)

    if isinstance(sigma, tuple):
        sigma = list(sigma)

    if not isinstance(sigma, (list, tuple)):
        raise_error(
            TypeError,
            f"Permutation ``sigma`` must be either a ``list`` or a ``tuple`` of ``int``s.",
        )

    nqubits = int(backend.ceil(backend.log2(len(sigma))))
    if sum([abs(s - i) for s, i in zip(sorted(sigma), range(2**nqubits))]) != 0:
        raise_error(
            ValueError, "Permutation sigma must contain all indices {0,...,n-1}"
        )

    if m > 0 and (m & (m - 1)) != 0:
        raise_error(ValueError, f"budget m must be a power‑of‑two")

    from qibo.quantum_info.utils import (  # pylint: disable=import-outside-toplevel
        decompose_permutation,
    )

    # factor sigma into the fewest layers such that
    # each layer has at most m disjoint transpositions, and
    # each layer moves a power‑of‑two number of indices
    layers = decompose_permutation(sigma, m)

    circuit = Circuit(nqubits)
    # in case we have more than one permutation to do, do it in layers
    for layer in layers:
        m = len(layer)
        ell, col_gates, A = _perm_column_ops(layer, nqubits, backend)
        row_gates = _perm_row_ops(A, ell, m, nqubits, backend)
        flip_gates = _perm_pair_flip_ops(nqubits, m, backend)
        col_row = col_gates + row_gates
        circuit.add(col_row)
        circuit.add(flip_gates)
        circuit.add(col_row[::-1])

    return circuit


def phase_encoder(data, rotation: str = "RY", backend=None, **kwargs):
    """Create circuit that performs the phase encoding of ``data``.

    Args:
        data (ndarray or list): :math:`1`-dimensional array of phases to be loaded.
        rotation (str, optional): If ``"RX"``, uses :class:`qibo.gates.gates.RX` as rotation.
            If ``"RY"``, uses :class:`qibo.gates.gates.RY` as rotation.
            If ``"RZ"``, uses :class:`qibo.gates.gates.RZ` as rotation.
            Defaults to ``"RY"``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in phase encoding.
    """
    if not isinstance(rotation, str):
        raise_error(
            TypeError,
            f"``rotation`` must be type str, but it is type {type(rotation)}.",
        )

    backend = _check_backend(backend)

    if isinstance(data, list):
        # TODO: Fix this mess with qibo native dtypes
        try:
            type_test = data[0].dtype
        except AttributeError:  # pragma: no cover
            type_test = type(data[0])

        data = backend.cast(data, dtype=type_test)

    if rotation not in ["RX", "RY", "RZ"]:
        raise_error(ValueError, f"``rotation`` {rotation} not found.")

    nqubits = len(data)
    gate = getattr(gates, rotation.upper())

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gate(qubit, 0.0) for qubit in range(nqubits))
    circuit.set_parameters(data)

    return circuit


def sparse_encoder(
    data, method: str = "li", nqubits: int = None, backend=None, **kwargs
):
    """Create circuit that encodes :math:`1`-dimensional data in a subset of amplitudes
    of the computational basis.

    Consider a sparse-access model, where for a data vector
    :math:`\\mathbf{x} \\in \\mathbb{C}^{d}`, with :math:`d = 2^{n}` and
    :math:`s` non-zero amplitudes, one has access to the data vector
    :math:`\\mathbf{y}` of the form

    .. math::
        \\mathbf{y} = \\left\\{ (b_{1}, x_{1}), \\, \\dots, \\, (b_{s}, x_{s}) \\right\\} \\, ,


    where :math:`\\{x_{j}\\}_{j\\in[s]}` is the non-zero components of :math:`\\mathbf{x}`
    and :math:`\\{b_{j}\\}_{j\\in[s]}` is the set of addresses associated with these values.
    Then, this function generates a quantum circuit  :math:`s\\text{-}\\mathrm{Load}` that encodes
    :math:`\\mathbf{x}` in the amplitudes of an :math:`n`-qubit quantum state as

    .. math::
        s\\text{-}\\mathrm{Load}(\\mathbf{y}) \\, \\ket{0}^{\\otimes \\, n} = \\sum_{j\\in[s]} \\,
            \\frac{x_{j}}{\\|\\mathbf{x}\\|_{2}} \\, \\ket{b_{j}} \\, ,

    where :math:`\\|\\cdot\\|_{2}` is the Euclidean norm.

    The resulting circuit parametrizes ``data`` in hyperspherical coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.


    Args:
        data (ndarray or list or zip): sequence of tuples of the form :math:`(b_{j}, x_{j})`.
            The addresses :math:`b_{j}` can be either integers or in bitstring
            format of size :math:`n`.
        method (str, optional): method to be used, either ``li`` or ``farias``. They refer to
            methods in references [1] and [2], respectively.
            Defaults to ``li``
        nqubits (int, optional): total number of qubits in the system.
            To be used when :math:`b_j` are integers. If :math:`b_j` are strings and
            ``nqubits`` is ``None``, defaults to the length of the strings :math:`b_{j}`.
            Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads sparse :math:`\\mathbf{x}`.

    References:
        1. L. Li, and J. Luo,
        *Nearly optimal circuit size for sparse quantum state preparation*
        `arXiv:2406.16142 (2024) <https://doi.org/10.48550/arXiv.2406.16142>`_.

        2. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*,
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.

        3. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.
    """
    backend = _check_backend(backend)

    if isinstance(data, zip):
        data = list(data)

    # TODO: Fix this mess with qibo native dtypes
    try:
        type_test = bool("int" in str(data[0][0].dtype))
    except AttributeError:
        type_test = bool("int" in str(type(data[0][0])))

    if type_test and nqubits is None:
        raise_error(
            ValueError,
            "``nqubits`` must be specified when computational basis states are "
            + "indidated by integers.",
        )

    if isinstance(data[0][0], str) and nqubits is None:
        nqubits = len(data[0][0])

    if method not in ("li", "farias"):
        raise_error(
            ValueError,
            f"``method`` should be either ``li`` or ``farias``, but it is {method}",
        )

    func = _sparse_encoder_farias if method == "farias" else _sparse_encoder_li

    return func(data, nqubits, backend, **kwargs)


def unary_encoder(data, architecture: str = "tree", backend=None, **kwargs):
    """Create circuit that performs the (deterministic) unary encoding of ``data``.

    Args:
        data (ndarray): :math:`1`-dimensional array of data to be loaded.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``diagonal``, uses a ladder-like structure.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in unary representation.
    """
    backend = _check_backend(backend)

    if isinstance(data, list):
        data = backend.cast(data, dtype=type(data[0]))

    if not isinstance(architecture, str):
        raise_error(
            TypeError,
            f"``architecture`` must be type str, but it is type {type(architecture)}.",
        )

    if architecture not in ["diagonal", "tree"]:
        raise_error(ValueError, f"``architecture`` {architecture} not found.")

    if architecture == "tree" and not math.log2(data.shape[0]).is_integer():
        raise_error(
            ValueError,
            "When ``architecture = 'tree'``, len(data) must be a power of 2. "
            + f"However, it is {len(data)}.",
        )

    nqubits = len(data)

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.X(nqubits - 1))
    circuit_rbs, _ = _generate_rbs_pairs(nqubits, architecture=architecture, **kwargs)
    circuit += circuit_rbs

    # calculating phases and setting circuit parameters
    phases = _generate_rbs_angles(data, architecture, nqubits, backend=backend)
    circuit.set_parameters(phases)

    return circuit


def unary_encoder_random_gaussian(
    nqubits: int, architecture: str = "tree", seed=None, backend=None, **kwargs
):
    """Create a circuit that performs the unary encoding of a random Gaussian state.

    At depth :math:`h` of the tree architecture, the angles :math:`\\theta_{k}
    \\in [0, 2\\pi]` of the the gates :math:`RBS(\\theta_{k})` are sampled from
    the following probability density function:

    .. math::
        p_{h}(\\theta) = \\frac{1}{2} \\, \\frac{\\Gamma(2^{h-1})}{\\Gamma^{2}(2^{h-2})} \\,
            \\left|\\sin(\\theta) \\, \\cos(\\theta)\\right|^{2^{h-1} - 1} \\, ,

    where :math:`\\Gamma(\\cdot)` is the
    `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.

    Args:
        nqubits (int): number of qubits.
        architecture(str, optional): circuit architecture used for the unary loader.
            If ``tree``, uses a binary-tree-based structure.
            Defaults to ``tree``.
        seed (int or :class:`numpy.random.Generator`, optional): Either a generator of
            random numbers or a fixed seed to initialize a generator. If ``None``,
            initializes a generator with a random seed. Defaults to ``None``.
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads a random Gaussian
        array in unary representation.

    References:
        1. A. Bouland, A. Dandapani, and A. Prakash, *A quantum spectral method for simulating
        stochastic processes, with applications to Monte Carlo*.
        `arXiv:2303.06719v1 [quant-ph] <https://arxiv.org/abs/2303.06719>`_
    """
    backend = _check_backend(backend)

    if not isinstance(nqubits, int):
        raise_error(
            TypeError, f"nqubits must be type int, but it is type {type(nqubits)}."
        )

    if nqubits <= 0.0:
        raise_error(
            ValueError, f"nqubits must be a positive integer, but it is {nqubits}."
        )

    if not isinstance(architecture, str):
        raise_error(
            TypeError,
            f"``architecture`` must be type str, but it is type {type(architecture)}.",
        )

    if architecture != "tree":
        raise_error(
            NotImplementedError,
            "Currently, this function only accepts ``architecture=='tree'``.",
        )

    if not math.log2(nqubits).is_integer():
        raise_error(ValueError, f"nqubits must be a power of 2, but it is {nqubits}.")

    if (
        seed is not None
        and not isinstance(seed, int)
        and not isinstance(seed, np.random.Generator)
    ):
        raise_error(
            TypeError, "seed must be either type int or numpy.random.Generator."
        )

    from qibo.quantum_info.random_ensembles import (  # pylint: disable=C0415
        _ProbabilityDistributionGaussianLoader,
    )

    # needs to rely on numpy's rng because of scipy
    local_state = (
        np.random.default_rng(seed) if seed is None or isinstance(seed, int) else seed
    )

    sampler = _ProbabilityDistributionGaussianLoader(
        a=0, b=2 * math.pi, seed=local_state
    )

    circuit = Circuit(nqubits, **kwargs)
    circuit.add(gates.X(nqubits - 1))
    circuit_rbs, pairs_rbs = _generate_rbs_pairs(nqubits, architecture, **kwargs)
    circuit += circuit_rbs

    phases = []
    for depth, row in enumerate(pairs_rbs, 1):
        phases.extend(sampler.rvs(depth=depth, size=len(row)))

    phases = backend.cast(phases, dtype=type(phases[0]))

    circuit.set_parameters(phases)

    return circuit


def up_to_k_hamming_weight_encoder(
    data,
    nqubits: int,
    up_to_k: int,
    codewords: List[int] = None,
    keep_antictrls: bool = False,
    backend: Optional[Backend] = None,
    **kwargs,
):
    """Create a circuit that encodes ``data`` in the Hamming-weight-:math:`\\leq k`
    subspace of ``nqubits``.

    Let :math:`\\mathbf{x}` be a :math:`1`-dimensional array of size

    .. math::
        d = \\sum_{l=0}^{k} \\binom{n}{l},

    and define the union-Hamming-weight-subspace

    .. math::
        B_{\\le k} \\equiv \\left\\{ \\ket{b_j} :
        b_j \\in \\{0,1\\}^{\\otimes n}, \\; |b_j| \\le k \\right\\},

    i.e., the set of all computational basis states of :math:`n` qubits whose
    bitstrings have Hamming weight less than or equal to :math:`k`.
    Equivalently,

    .. math::
        B_{\\le k} = \\bigcup_{w=0}^{k} B_w,

    where :math:`B_w` denotes the set of basis states of fixed Hamming weight
    :math:`w`.

    An amplitude encoder in the basis :math:`B_{\\le k}` is an
    :math:`n`-qubit parameterized quantum circuit
    :math:`\\operatorname{Load}_{B_{\\le k}}` such that

    .. math::
        \\operatorname{Load}_{B_{\\le k}}(\\mathbf{x}) \\, \\ket{0}^{\\otimes n}
        =
        \\frac{1}{\\|\\mathbf{x}\\|}
        \\sum_{j=1}^{d} x_j \\ket{b_j},

    where :math:`\\{ \\ket{b_j} \\}_{j=1}^d` is an enumeration of the elements
    of :math:`B_{\\le k}`.

    Resulting circuit parametrizes ``data`` in either ``hyperspherical`` or ``hopf`` coordinates
    in the :math:`(2^{n} - 1)`-unit sphere.

    Args:
        data (ndarray): :math:`1`-dimensional array of length
            :math:`d = \\sum_{l=0}^{k} \\binom{n}{l}` to be loaded in the
            amplitudes of a :math:`n`-qubit quantum state.
        nqubits (int): total number of qubits in the system.
        up_to_k (int): upper limit for the Hamming weight of the union-Hamming-weight-subspace
            in which the data to be loaded will be supported.
        codewords (list, optional): List of codewords used to encode the data in the given order.
            If ``None``, the codewords are set by the erhlich algorithm.
        keep_antictrls (bool, optional): If ``True`` and parametrization is ``hyperspherical``, we
            don't simplify the anti-controls when placing the RBS gates. For details, see [1].
        backend (:class:`qibo.backends.abstract.Backend`, optional): backend to be used
            in the execution. If ``None``, it uses the current backend. Defaults to ``None``.
        kwargs (dict, optional): Additional arguments used to initialize a Circuit object.
            For details, see the documentation of :class:`qibo.models.circuit.Circuit`.

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit that loads ``data`` in up-to-k encoding.

    References:
        1. R. M. S. Farias, T. O. Maciel, G. Camilo, R. Lin, S. Ramos-Calderer, and L. Aolita,
        *Quantum encoder for fixed-Hamming-weight subspaces*
        `Phys. Rev. Applied 23, 044014 (2025) <https://doi.org/10.1103/PhysRevApplied.23.044014>`_.

        2. `Hyperpherical coordinates <https://en.wikipedia.org/wiki/N-sphere>`_.
    """
    backend = _check_backend(backend)

    complex_data = bool("complex" in str(data.dtype))

    return _up_to_k_encoder_hyperspherical(
        data,
        nqubits,
        up_to_k,
        complex_data=complex_data,
        backend=backend,
        codewords=codewords,
        keep_antictrls=keep_antictrls,
        **kwargs,
    )
