Superoperator Transformations
=============================

Functions used to convert superoperators among their possible representations.
For more in-depth theoretical description of the representations and transformations,
we direct the reader to
`Wood, Biamonte, and Cory, Quant. Inf. Comp. 15, 0579-0811 (2015) <https://arxiv.org/abs/1111.6950>`_.


Vectorization
-------------

.. autofunction:: qibo.quantum_info.vectorization

.. note::
    Due to ``numpy`` limitations on handling transposition of tensors,
    this function will not work when the number of qubits :math:`n`
    is such that :math:`n > 16`.


Unvectorization
---------------

.. autofunction:: qibo.quantum_info.unvectorization

.. note::
    Due to ``numpy`` limitations on handling transposition of tensors,
    this function will not work when the number of qubits :math:`n`
    is such that :math:`n > 16`.


To Choi
-------

.. autofunction:: qibo.quantum_info.to_choi


To Liouville
------------

.. autofunction:: qibo.quantum_info.to_liouville


To Pauli-Liouville
------------------

.. autofunction:: qibo.quantum_info.to_pauli_liouville


To Chi
-------

.. autofunction:: qibo.quantum_info.to_chi


To Stinespring
--------------

.. autofunction:: qibo.quantum_info.to_stinespring


Choi to Liouville
-----------------

.. autofunction:: qibo.quantum_info.choi_to_liouville


Choi to Pauli-Liouville
-----------------------

.. autofunction:: qibo.quantum_info.choi_to_pauli


Choi to Kraus
-------------

.. autofunction:: qibo.quantum_info.superoperator_transformations.choi_to_kraus

.. note::
    Due to the spectral decomposition subroutine in this function, the resulting Kraus
    operators :math:`\{K_{\alpha}\}_{\alpha}` might contain global phases. That
    implies these operators are not exactly equal to the "true" Kraus operators
    :math:`\{K_{\alpha}^{(\text{ideal})}\}_{\alpha}`. However, since these are
    global phases, the operators' actions are the same, i.e.

    .. math::
        K_{\alpha} \, \rho \, K_{\alpha}^{\dagger} = K_{\alpha}^{\text{(ideal)}} \, \rho \,\,
            (K_{\alpha}^{\text{(ideal)}})^{\dagger} \,\,\,\,\, , \,\, \forall \, \alpha

.. note::
    User can set ``validate_cp=False`` in order to speed up execution by not checking if
    input map ``choi_super_op`` is completely positive (CP) and Hermitian. However, that may
    lead to erroneous outputs if ``choi_super_op`` is not guaranteed to be CP. We advise users
    to either set this flag carefully or leave it in its default setting (``validate_cp=True``).


Choi to Chi-matrix
------------------

.. autofunction:: qibo.quantum_info.choi_to_chi


Choi to Stinespring
-------------------

.. autofunction:: qibo.quantum_info.choi_to_stinespring


Kraus to Choi
-------------

.. autofunction:: qibo.quantum_info.kraus_to_choi


Kraus to Liouville
------------------

.. autofunction:: qibo.quantum_info.kraus_to_liouville


Kraus to Pauli-Liouville
------------------------

.. autofunction:: qibo.quantum_info.kraus_to_pauli


Kraus to Chi-matrix
-------------------

.. autofunction:: qibo.quantum_info.kraus_to_chi


Kraus to Stinespring
--------------------

.. autofunction:: qibo.quantum_info.kraus_to_stinespring


Liouville to Choi
-----------------

.. autofunction:: qibo.quantum_info.liouville_to_choi


Liouville to Pauli-Liouville
----------------------------

.. autofunction:: qibo.quantum_info.liouville_to_pauli


Liouville to Kraus
------------------

.. autofunction:: qibo.quantum_info.liouville_to_kraus

.. note::
    Due to the spectral decomposition subroutine in this function, the resulting Kraus
    operators :math:`\{K_{\alpha}\}_{\alpha}` might contain global phases. That
    implies these operators are not exactly equal to the "true" Kraus operators
    :math:`\{K_{\alpha}^{(\text{ideal})}\}_{\alpha}`. However, since these are
    global phases, the operators' actions are the same, i.e.

    .. math::
        K_{\alpha} \, \rho \, K_{\alpha}^{\dagger} = K_{\alpha}^{\text{(ideal)}} \, \rho \,\,
            (K_{\alpha}^{\text{(ideal)}})^{\dagger} \,\,\,\,\, , \,\, \forall \, \alpha


Liouville to Chi-matrix
-----------------------

.. autofunction:: qibo.quantum_info.liouville_to_chi


Liouville to Stinespring
------------------------

.. autofunction:: qibo.quantum_info.liouville_to_stinespring


Pauli-Liouville to Liouville
----------------------------

.. autofunction:: qibo.quantum_info.pauli_to_liouville


Pauli-Liouville to Choi
-----------------------

.. autofunction:: qibo.quantum_info.pauli_to_choi



Pauli-Liouville to Kraus
------------------------

.. autofunction:: qibo.quantum_info.pauli_to_kraus


Pauli-Liouville to Chi-matrix
-----------------------------

.. autofunction:: qibo.quantum_info.pauli_to_chi


Pauli-Liouville to Stinespring
------------------------------

.. autofunction:: qibo.quantum_info.pauli_to_stinespring


Chi-matrix to Choi
------------------

.. autofunction:: qibo.quantum_info.chi_to_choi


Chi-matrix to Liouville
-----------------------

.. autofunction:: qibo.quantum_info.chi_to_liouville


Chi-matrix to Pauli-Liouville
-----------------------------

.. autofunction:: qibo.quantum_info.chi_to_pauli


Chi-matrix to Kraus
-------------------

.. autofunction:: qibo.quantum_info.chi_to_kraus

.. note::
    Due to the spectral decomposition subroutine in this function, the resulting Kraus
    operators :math:`\{K_{\alpha}\}_{\alpha}` might contain global phases. That
    implies these operators are not exactly equal to the "true" Kraus operators
    :math:`\{K_{\alpha}^{(\text{ideal})}\}_{\alpha}`. However, since these are
    global phases, the operators' actions are the same, i.e.

    .. math::
        K_{\alpha} \, \rho \, K_{\alpha}^{\dagger} = K_{\alpha}^{\text{(ideal)}} \, \rho \,\,
            (K_{\alpha}^{\text{(ideal)}})^{\dagger} \,\,\,\,\, , \,\, \forall \, \alpha

.. note::
    User can set ``validate_cp=False`` in order to speed up execution by not checking if
    the Choi representation obtained from the input ``chi_matrix`` is completely positive
    (CP) and Hermitian. However, that may lead to erroneous outputs if ``choi_super_op``
    is not guaranteed to be CP. We advise users to either set this flag carefully or leave
    it in its default setting (``validate_cp=True``).


Chi-matrix to Stinespring
-------------------------

.. autofunction:: qibo.quantum_info.chi_to_stinespring


Stinespring to Choi
-------------------

.. autofunction:: qibo.quantum_info.stinespring_to_choi


Stinespring to Liouville
------------------------

.. autofunction:: qibo.quantum_info.stinespring_to_liouville


Stinespring to Pauli-Liouville
------------------------------

.. autofunction:: qibo.quantum_info.stinespring_to_pauli


Stinespring to Kraus
--------------------

.. autofunction:: qibo.quantum_info.stinespring_to_kraus


Stinespring to Chi-matrix
-------------------------

.. autofunction:: qibo.quantum_info.stinespring_to_chi
