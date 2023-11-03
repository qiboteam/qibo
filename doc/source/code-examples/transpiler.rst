.. _tutorials_transpiler:

How to modify the transpiler?
=============================

The transpiler is responsible for transforming a circuit to respect the chip connectivity and native gates.
The user can modify these attributes before executing a circuit.
Multiple transpilation steps can be implemented using the :class:`qibo.transpiler.pipeline.Pipeline`:
.. testcode:: python

    from qibo.transpiler.pipeline import Passes
    from qibo.transpiler.abstract import NativeType
    from qibo.transpiler.star_connectivity import StarConnectivity
    from qibo.transpiler.unroller import NativeGates

    transpiler_pipeline = Passes(
        [
            StarConnectivity(middle_qubit=2),
            NativeGates(two_qubit_natives=NativeType.CZ),
        ]
    )

In this case circuits will first be transpiled to respect the 5-qubit star connectivity, with qubit 2 as the middle qubit. This will potentially add some SWAP gates. Then all gates will be converted to native.
The :class:`qibo.transpiler.unroller.NativeGates` transpiler used in this example assumes Z, RZ, GPI2 or U3 as the single-qubit native gates, and supports CZ and iSWAP as two-qubit natives.
In this case we restricted the two-qubit gate set to CZ only.
If the circuit to be executed contains gates that are not included in this gate set, they will be transformed to multiple gates from the gate set.
Arbitrary single-qubit gates are typically transformed to U3.
Arbitrary two-qubit gates are transformed to two or three CZ gates following their `universal CNOT decomposition <https://arxiv.org/abs/quant-ph/0307177>`_.
The decomposition of some common gates such as the SWAP and CNOT is hard-coded for efficiency.
