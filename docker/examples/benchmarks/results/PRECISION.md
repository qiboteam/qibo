# Simulation precision

Qibo allows the user to easily switch between single (``complex64``)
and double (``complex128``) precision as described in
[How to modify the simulation precision?](https://qibo.readthedocs.io/en/latest/examples.html#how-to-modify-the-simulation-precision).
In this section we compare simulation performance of both precisions.
We find that as the number of qubits grows using single precision is ~2
times faster on GPU and ~1.5 faster on CPU.


`nqubits` | CPU c64 | CPU c128 | GPU c64 | GPU c128
-- | -- | -- | -- | --
23 | 0.41793 | 0.57448 | 0.07276 | 0.09249
24 | 0.83167 | 1.28207 | 0.09982 | 0.14664
25 | 1.75593 | 2.68645 | 0.15088 | 0.25463
26 | 3.68007 | 5.60514 | 0.26269 | 0.47597
27 | 7.71366 | 11.78381 | 0.48103 | 0.93437
28 | 16.36245 | 24.84334 | 0.93884 | 1.87903
29 | 34.66599 | 52.47754 | 1.88652 | 3.8849
30 | 73.43444 | 110.6095 | 3.90816 | 8.10977
31 | 155.01974 | 232.35723 | 8.08614 |        
32 | 329.56916 | 488.00321 |         |        
33 | 694.67844 | 1021.06619 |         |        
34 | 1465.81645 |         |         |  

![precision-benchmarks](../images/qibo_c64_vs_c128.png)
