.. _installation:

************
Installation
************

.. note::

    Python 3.6 or higher is required!

PyPi
====

.. code-block:: bash

    pip install ceml

.. note::
    The package hosted on PyPI uses the cpu only. If you want to use the gpu, you have to install CEML manually - see next section.


Git
===
Download or clone the repository:

.. code:: bash

    git clone https://github.com/andreArtelt/ceml.git
    cd ceml

Install all requirements (listed in ``requirements.txt``):

.. code:: bash

    pip install -r requirements.txt

.. note::
    If you want to use a gpu/tpu, you have to install the gpu version of jax, tensorflow and PyTorch manually.
    
    Do not use ``pip install -r requirements.txt``

Install the toolbox itself:

.. code:: bash

    pip install
