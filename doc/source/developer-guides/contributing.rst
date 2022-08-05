How to contribute?
==================

In the following paragraphs we describe the guidelines for contributing to Qibo.

Code review process
-------------------

All code submissions require a review and continous integration tests
beforing accepting the code and merging to the git master branch.

We use the GitHub pull request mechanism which can be summarized as follows:

1. Fork the Qibo repository.

2. Checkout master and create a new branch from it

    .. code-block::

        git checkout master -b new_branch

   where ``new_branch`` is the name of your new branch.

3. Implement your new feature on ``new_branch``.

4. When you are done, push your branch with:

    .. code-block::

        git push origin new_branch

5. At this point you can create a pull request by visiting the Qibo GitHub page.

6. The review process will start and changes in your code may be requested.

Tests
-----

When commits are pushed to the branches in the GitHub repository,
we perform integrity checks to ensure that the new features do
not break Qibo functionalities and meets our coding standards.

The current code standards that are applied to any new changes:

- **Tests**: We use pytest to run our tests that must continue to pass when new changes are integrated in the code. Regression tests, which are run by the continuous integration workflow are stored in ``qibo/tests``. These tests contain several examples about how to use Qibo.
- **Coverage**: Test coverage should be maintained / be at least at the same level when new features are implemented.
- **Pylint**: Test code for anomalies, such as bad coding practices, missing documentation, unused variables.
- **Pre commit**: We use pre-commit to enforce automation and to format the code. The `pre-commit ci <https://pre-commit.ci/>`_ will automatically run pre-commit whenever a commit is performed inside a pull request.

Besides the linter, further custom rules are applied e.g. checks for ``print`` statements that bypass the logging system
(such check can be excluded line by line with the ``# CodeText:skip`` flag).

Documentation
-------------

The Qibo documentation is automatically generated with `sphinx
<https://www.sphinx-doc.org/>`_, thus all functions should be documented using
docstrings. The ``doc`` folder contains the project setup for the documentation
web page.

The documentation requirements can be installed with:

.. code-block::

    pip install qibo[docs]

Alternatively, install the packages listed in the ``extras_require`` option in
``setup.py``.

In order to build the documentation web page locally please perform the following steps:

.. code-block::

    cd doc
    make html

This last command generates a web page in ``doc/build/html/``. You can browse
the local compiled documentation by opening ``doc/build/html/index.html``.

The sections in the documentation are controlled by the ``*.rst`` files located
in ``doc/source/``. The application tutorials are rendered from markdown by
linking the respective files from ``examples/`` in ``doc/source/tutorials/``.
