[1mdiff --git a/src/qibo/error_mitigation/abstract.py b/src/qibo/error_mitigation/abstract.py[m
[1mindex e77434bd7..e23df5bed 100644[m
[1m--- a/src/qibo/error_mitigation/abstract.py[m
[1m+++ b/src/qibo/error_mitigation/abstract.py[m
[36m@@ -70,7 +70,7 @@[m [mclass MitigatedMeasurementResult(MeasurementResult):[m
     pass[m
 [m
 [m
[31m-@dataclass[m
[32m+[m[32m@dataclass(frozen=True)[m
 class ErrorMitigationRoutine(ABC):[m
 [m
     circuit: Optional[Circuit] = None[m
[36m@@ -148,7 +148,7 @@[m [mclass ErrorMitigationRoutine(ABC):[m
         return new_circuits[m
 [m
 [m
[31m-@dataclass[m
[32m+[m[32m@dataclass(frozen=True)[m
 class DataRegressionErrorMitigation(ErrorMitigationRoutine):[m
 [m
     n_training_samples: Optional[int] = 50[m
[1mdiff --git a/src/qibo/error_mitigation/cdr.py b/src/qibo/error_mitigation/cdr.py[m
[1mindex 72533a245..30cafd8fe 100644[m
[1m--- a/src/qibo/error_mitigation/cdr.py[m
[1m+++ b/src/qibo/error_mitigation/cdr.py[m
[36m@@ -14,7 +14,7 @@[m [mfrom qibo.gates.abstract import Gate, ParametrizedGate[m
 NPBACKEND = NumpyBackend()[m
 [m
 [m
[31m-@dataclass[m
[32m+[m[32m@dataclass(frozen=True)[m
 class CDR(DataRegressionErrorMitigation):[m
 [m
     replacement_gates: Optional[Tuple[Gate, List[Dict]]] = None[m
[1mdiff --git a/src/qibo/error_mitigation/zne.py b/src/qibo/error_mitigation/zne.py[m
[1mindex 8f9377964..df7023fc9 100644[m
[1m--- a/src/qibo/error_mitigation/zne.py[m
[1m+++ b/src/qibo/error_mitigation/zne.py[m
[36m@@ -4,6 +4,7 @@[m [mfrom functools import cache, cached_property[m
 from typing import Iterable, List, Optional, Tuple[m
 [m
 import numpy as np[m
[32m+[m[32mfrom numpy.typing import ArrayLike[m
 [m
 from qibo import Circuit, gates[m
 from qibo.config import raise_error[m
[36m@@ -12,7 +13,7 @@[m [mfrom qibo.hamiltonians.abstract import AbstractHamiltonian[m
 from qibo.noise import NoiseModel[m
 [m
 [m
[31m-@dataclass[m
[32m+[m[32m@dataclass(frozen=True)[m
 class ZNE(ErrorMitigationRoutine):[m
 [m
     noise_levels: Iterable[int] = tuple(range(4))[m
[36m@@ -125,6 +126,18 @@[m [mclass ZNE(ErrorMitigationRoutine):[m
             for num_insertions in self.noise_levels[m
         ][m
 [m
[32m+[m[32m    @cache[m
[32m+[m[32m    def noisy_expectations(self, nshots) -> ArrayLike:[m
[32m+[m[32m        exp_vals = [[m
[32m+[m[32m            self.observable.expectation([m
[32m+[m[32m                circ, nshots=nshots, readout_mitigation=self.readout_mitigation[m
[32m+[m[32m            )[m
[32m+[m[32m            for circ in self.noisy_circuits[m
[32m+[m[32m        ][m
[32m+[m[32m        return self.backend.np.stack(exp_vals)[m
[32m+[m
[32m+[m[41m        [m
[32m+[m
     def __call__([m
         self,[m
         circuit: Optional[Circuit] = None,[m
[36m@@ -135,6 +148,7 @@[m [mclass ZNE(ErrorMitigationRoutine):[m
         noisy_circuits = self._noisy_circuits(circuit)[m
         noisy_circuits = self._circuit_preprocessing(noisy_circuits, noise_model)[m
         observable = self._observable(observable)[m
[32m+[m[32m        breakpoint()[m
         exp_vals = [[m
             observable.expectation([m
                 circ, nshots=nshots, readout_mitigation=self.readout_mitigation[m
