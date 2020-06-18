Installing QIBO
===============

The QIBO package comes with the following modules:

* :ref:`installing-with-pip`
* :ref:`installing-from-source`

_______________________

.. _installing-with-pip:

Installing with pip
-------------------

Make sure you have Python 3.6 or greater

.. code-block:: bash

      pip install qibo

.. _installing-from-source:

Installing from source
----------------------

In order to install you can simply clone this repository with

.. code-block:: bash

      git clone git@github.com:Quantum-TII/qibo.git

and then proceed with the installation with:

.. code-block:: bash

      python setup.py install

If you prefer to keep changes always synchronized with the code then install using the develop option:

.. code-block:: bash

      python setup.py develop