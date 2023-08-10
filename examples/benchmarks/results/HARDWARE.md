# Hardware configurations

A core point in Qibo is the support of different hardware configurations
despite its simple installation procedure. The user can easily switch between
CPU and GPU as described in
[How to select hardware devices?](https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-select-hardware-devices).
A question that arises is how to determine the optimal device configuration for
the circuit one would like to simulate.
While the answer to this question depends both on the circuit specifics
(number of qubits, number of gates, number of re-executions) and the
exact hardware specifications (CPU or GPU model and available memory), here
we provide a basic comparison using the DGX station to simulate the QFT circuit
in double precision (`complex128`).


## Heuristic rules

Based on the results presented below, in the following table we provide some
heuristic rules for optimal device selection according to the number of qubits.
More stars means a shorter execution time is expected.
We stress that these general rules may not be valid on every case as the optimal
configuration depends on various factors, such as the exact circuit structure
and hardware specifications (CPU and GPU speed and memory).

`nqubits` | 0-15 | 15-30 | >30
-- | :--: | :--: | :--:
CPU single thread | `***` | `*` | `*`
CPU multi-threading | `*` | `**` | `**`
single GPU | `*` | `***` | `**`
multi-GPU | - | - | `***`


## Large circuits

Here we measure QFT execution time for large circuits (25 to 33 qubits) using
different CPU thread configuration, single GPU and multi-GPU.

`nqubits` | single-GPU | multi-GPU | 1-thread | 10-threads | 20-threads | 40-threads
-- | -- | -- | -- | -- | -- | --
25 | 0.25463 | 2.06511 | 31.42569 | 4.29928 | 2.83574 | 2.68645
26 | 0.47597 | 2.7348 | 67.51781 | 8.95446 | 5.86079 | 5.60514
27 | 0.93437 | 3.86462 | 144.42056 | 18.79864 | 12.2377 | 11.78381
28 | 1.87903 | 6.34791 | 309.06432 | 40.01433 | 25.70007 | 24.84334
29 | 3.8849 | 11.05362 | 663.96249 | 84.17542 | 53.86609 | 52.47754
30 | 8.10977 | 21.21188 | 1426.46899 | 177.51885 | 113.43018 | 110.6095
31 | 68.09852 | 41.26969 | 3064.75608 | 375.57098 | 238.23101 | 232.35723
32 | 182.78114 | 66.40625 | 6584.73906 | 794.48593 | 501.6703 | 488.00321
33 | 468.13457 | 161.33294 | 14147.73073 | 1667.25349 | 1046.75898 | 1021.06619

![hardware-large](../images/qibo_configurations.png)


## Small circuits

Here we measure QFT execution time for small circuits (25 to 33 qubits) using
different single-thread CPU, multi-thread CPU and GPU.

`nqubits` | single-GPU | 1-thread | 40-threads
-- | -- | -- | --
5 | 0.00348 | 0.0021 | 0.00225
6 | 0.0047 | 0.00284 | 0.00305
7 | 0.00594 | 0.0036 | 0.00755
8 | 0.00747 | 0.00455 | 0.01225
9 | 0.00869 | 0.00557 | 0.01734
10 | 0.01048 | 0.00683 | 0.02044
11 | 0.0123 | 0.00827 | 0.02346
12 | 0.01439 | 0.01043 | 0.02695
13 | 0.01653 | 0.01299 | 0.02982
14 | 0.01975 | 0.01885 | 0.03459
15 | 0.02168 | 0.02576 | 0.03783
16 | 0.02439 | 0.04158 | 0.04161
17 | 0.02723 | 0.07478 | 0.04668
18 | 0.03017 | 0.14607 | 0.05237
19 | 0.03361 | 0.29966 | 0.06529
20 | 0.03881 | 0.63581 | 0.09226
21 | 0.04525 | 1.37577 | 0.1233
22 | 0.0682 | 3.08223 | 0.23931
23 | 0.09557 | 6.77251 | 0.56214

![hardware-small](../images/qibo_configurations_small.png)
