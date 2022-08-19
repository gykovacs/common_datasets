Examples
********

Loading a dataset
=================

.. code-block:: python

    # binary classification
    import mldatasets.binary_classification as binclas

    dataset = binclas.load_abalone19()

    # multiclass classification
    import mldatasets.multiclass_classification as multclas

    dataset = multclas.load_abalone()

    # regression
    from mldatasets import regression

    dataset = regression.load_treasury()

Querying all dataset loaders and loading a dataset
==================================================

.. code-block:: python

    # binary classification
    import mldatasets.binary_classification as binclas

    data_loaders = binclas.get_data_loaders()

    dataset_0 = data_loaders[0]()

    # multiclass classification
    import mldatasets.multiclass_classification as multclas

    data_loaders = multclas.get_data_loaders()

    dataset_0 = data_loaders[0]()

    # regression
    from mldatasets import regression

    data_loaders = regression.get_data_loaders()

    dataset_0 = data_loaders[0]()

Querying the loaders of the 5 smallest datasets regarding the total number of records
=====================================================================================

.. code-block:: python

    # binary classification
    import mldatasets.binary_classification as binclas

    data_loaders = binclas.get_filtered_data_loaders(n_smallest=5, sorting='n')

    dataset_0 = data_loaders[0]()

    # multiclass classification
    import mldatasets.multiclass_classification as multclas

    data_loaders = multclas.get_data_loaders(n_smallest=5, sorting='n')

    dataset_0 = data_loaders[0]()

    # regression
    from mldatasets import regression

    data_loaders = regression.get_data_loaders(n_smallest=5, sorting='n')

    dataset_0 = data_loaders[0]()
