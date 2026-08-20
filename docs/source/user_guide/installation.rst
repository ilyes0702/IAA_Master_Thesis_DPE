Installation
========================

You can install the latest development version or specific releases of SeqControl directly from its GitHub repository.

Installing with pip
-------------------

The easiest way to install directly from GitHub is using ``pip``.

Latest Development Version (Main Branch)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the latest code directly from the ``main`` branch, run:

.. code-block:: bash

    pip install git+https://github.com/ilyes0702/.git

Installing a Specific Branch or Tag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to install a specific version, release tag, or branch, append ``@branch-or-tag-name`` 
to the end of the URL:

.. code-block:: bash

    # Install from a specific branch
    pip install git+https://github.com/your-username/your-repo-name.git@dev

    # Install from a specific tag or release
    pip install git+https://github.com/your-username/your-repo-name.git@v1.0.0

Editable / Developer Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you plan to modify the package source code or contribute to development:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name

    # Install in editable mode
    pip install -e .

Including Optional Dependencies
-------------------------------

If your package defines optional extras (e.g., ``[dev]`` or ``[docs]`` in ``setup.py`` or ``pyproject.toml``), 
you can install them by wrapping the package name:

.. code-block:: bash

    pip install "your-package-name[dev] @ git+https://github.com/your-username/your-repo-name.git"

Using Environment Files
-----------------------

requirements.txt
^^^^^^^^^^^^^^^^

To list this GitHub dependency inside a ``requirements.txt`` file, add the following line:

.. code-block:: text

    your-package-name @ git+https://github.com/your-username/your-repo-name.git@v1.0.0

Conda environment.yml
^^^^^^^^^^^^^^^^^^^^^

To include this package in a Conda ``environment.yml`` file, add it under the ``pip`` section:

.. code-block:: yaml

    name: my_env
    channels:
      - defaults
    dependencies:
      - python=3.10
      - pip:
        - git+https://github.com/your-username/your-repo-name.git@main

Prerequisites & Options
-----------------------

.. note::
   * **Git Requirement:** Ensure ``git`` is installed on your local system and added to your system path.
   * **Private Repositories:** For private GitHub repositories, ensure your local SSH keys or Personal Access Tokens (PAT) are configured, and use the SSH format:

   .. code-block:: bash

       pip install git+ssh://git@github.com/your-username/your-repo-name.git