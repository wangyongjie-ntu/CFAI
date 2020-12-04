.. _installation:

************
Installation
************

.. note::

    Python 3+, Pytorch, Scikit-learn are required!

PyPi
====

.. code-block:: bash

    pip install ceml

.. note::
    The package hosted on PyPI uses the cpu only. For GPU requirements, please modify the source code accordingly.


Git
===
Download the repository:

.. code:: bash

    git clone https://github.com/wangyongjie-ntu/Counterfactual-Explanations-Pytorch
    cd ceml
    pip install -r requirements.txt
    pip install -e .
