|CircleCI|_ |GitHub|_ |Codecov|_ |ReadTheDocs|_ |PythonVersion|_ |pylint|_ |PyPi|_

.. |CircleCI| image:: https://circleci.com/gh/gykovacs/common_datasets.svg?style=svg
.. _CircleCI: https://circleci.com/gh/gykovacs/common_datasets

.. |GitHub| image:: https://github.com/gykovacs/common_datasets/workflows/Python%20package/badge.svg?branch=master
.. _GitHub: https://github.com/gykovacs/common_datasets/workflows/Python%20package/badge.svg?branch=master

.. |Codecov| image:: https://codecov.io/gh/gykovacs/common_datasets/branch/master/graph/badge.svg?token=GQNNasvi4z
.. _Codecov: https://codecov.io/gh/gykovacs/common_datasets

.. |ReadTheDocs| image:: https://readthedocs.org/projects/common_datasets/badge/?version=latest
.. _ReadTheDocs: https://common_datasets.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-brightgreen
.. _PythonVersion: https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-brightgreen

.. |pylint| image:: https://img.shields.io/badge/pylint-10.0-green
.. _pylint: https://img.shields.io/badge/pylint-10.0-green

.. |PyPi| image:: https://badge.fury.io/py/common_datasets.svg
.. _PyPi: https://badge.fury.io/py/common_datasets


common-datasets: common machine learning datasets
#################################################

This package provides an unofficial collection of datasets widely used in the evaluation of machine learning
techniques, mainly small and imbalanced datasets for binary, multiclass classification and regression. The
datasets are provided in the usual `sklearn.datasets` format, with missing data imputation and the encoding
of category and ordinal features. The authors of this repository do not own any licenses for the datasets,
the goal of the project is to provide a stanardized collection of datasets for research purposes.

PLEASE DO NOT CITE OR REFER TO THIS PACKAGE IN ANY FORM!

If you use data through this repository, please cite the original works publishing and specifying these datasets:

.. code-block:: BibTex

  @article{keel,
    author={Alcala-Fdez, J. and Fernandez, A. and Luengo, J. and Derrac, J. and Garcia, S.
            and Sanchez, L. and Herrera, F.},
    title={KEEL Data-Mining Software Tool: Data Set Repository, Integration of Algorithms
            and Experimental Analysis Framework},
    journal={Journal of Multiple-Valued Logic and Soft Computing},
    volume={17},
    number={2-3},
    year={2011},
    pages={255-287}}

  @misc{uci,
    author = "Dua, Dheeru and Karra Taniskidou, Efi",
    year = "2017",
    title = "{UCI} Machine Learning Repository",
    url = "http://archive.ics.uci.edu/ml",
    institution = "University of California, Irvine, School of Information and Computer Sciences"}

  @article{krnn,
    author={X. J. Zhang and Z. Tari and M. Cheriet},
    title={{KRNN}: k {Rare-class Nearest Neighbor} classification},
    journal={Pattern Recognition},
    year={2017},
    volume={62},
    number={2},
    pages={33--44}
    }

For each individual dataset the citation key referring to its publisher or a relevant publication
in which the dataset in the given configuration has been used is provided as part of the dataset.
For example:

.. code-block:: python

  # binary classification
  >> import common_datasets.binary_classification as binclas

  >> dataset = bin_clas.load_abalone19()
  >> dataset['citation_key']
  'keel'

Introduction
************

The package contains 119 binary classification, 23 multiclass classification and 23 regression datasets.


Installation
************

The package can be cloned from GitHub in the usual way, and the latest stable version is also available in the PyPI repository:

.. code-block:: bash

  pip install common_datasets

Use cases
*********

Loading a dataset
=================

.. code-block:: python

  # binary classification
  import common_datasets.binary_classification as binclas

  dataset = binclas.load_abalone19()

  # multiclass classification
  import common_datasets.multiclass_classification as multclas

  dataset = multclas.load_abalone()

  # regression
  from common_datasets import regression

  dataset = regression.load_treasury()

Querying all dataset loaders and loading a dataset
==================================================

.. code-block:: python

  # binary classification
  import common_datasets.binary_classification as binclas

  data_loaders = binclas.get_data_loaders()

  dataset_0 = data_loaders[0]()

  # multiclass classification
  import common_datasets.multiclass_classification as multclas

  data_loaders = multclas.get_data_loaders()

  dataset_0 = data_loaders[0]()

  # regression
  from common_datasets import regression

  data_loaders = regression.get_data_loaders()

  dataset_0 = data_loaders[0]()

Querying the loaders of the 5 smallest datasets regarding the total number of records
=====================================================================================

.. code-block:: python

  # binary classification
  import common_datasets.binary_classification as binclas

  data_loaders = binclas.get_filtered_data_loaders(n_smallest=5, sorting='n')

  dataset_0 = data_loaders[0]()

  # multiclass classification
  import common_datasets.multiclass_classification as multclas

  data_loaders = multclas.get_data_loaders(n_smallest=5, sorting='n')

  dataset_0 = data_loaders[0]()

  # regression
  from common_datasets import regression

  data_loaders = regression.get_data_loaders(n_smallest=5, sorting='n')

  dataset_0 = data_loaders[0]()


Documentation
*************

* For a detailed documentation and parameters of the functions see http://common_datasets.readthedocs.io.
