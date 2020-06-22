How to contribute
=================

In the following paragraphs we describe the guidelines for contributing to QIBO.

Code review process
-------------------

All code submissions require a review and continous integration tests
beforing accepting the code and merging to the git master branch.

We use the GitHub pull request mechanism which can be summarized as follows:

1. Fork the QIBO repository.

2. Checkout master and create a new branch from it

    .. code-block::

        git checkout master -b new_branch

   where ``new_branch`` is the name of your new branch.

3. Implement your new feature on ``new_branch``.

4. When you are done, push your branch with:

    .. code-block::

        git push origin new_branch

5. At this point you can create a pull request by visiting the QIBO GitHub page.

6. The review process will start and changes in your code may be requested.

Tests
-----

When commits are pushed to the branches in the GitHub repository,
we perform integrity checks to ensure that the new features do
not break QIBO functionalities and meets our coding standards.

Where the current code standards that are applied to any new changes:

- **Tests**: We use pytest to run our tests thus tests must continues to pass when new changes are integrated in the code.
- **Coverage**: Code should be at least mantain code coverage when new features are implemented.