"""
Testing multiclass arff datasets.
"""

import pytest

from sklearn.tree import DecisionTreeClassifier

import mldb.multiclass_classification as mult_clas

loaders = [mult_clas.load_automobile,
            mult_clas.load_balance,
            mult_clas.load_car,
            mult_clas.load_cleveland,
            mult_clas.load_contraceptive,
            mult_clas.load_dermatology,
            mult_clas.load_fars,
            mult_clas.load_flare,
            mult_clas.load_hayes_roth,
            mult_clas.load_kr_vs_k,
            mult_clas.load_led7digit,
            mult_clas.load_movement_libras,
            mult_clas.load_newthyroid,
            mult_clas.load_nursery,
            mult_clas.load_page_blocks,
            mult_clas.load_post_operative,
            mult_clas.load_segment,
            mult_clas.load_splice,
            mult_clas.load_tae,
            mult_clas.load_vowel,
            mult_clas.load_zoo,
            mult_clas.load_glass,
            mult_clas.load_satimage,
            mult_clas.load_ecoli,
            mult_clas.load_abalone,
            mult_clas.load_yeast]

@pytest.mark.parametrize("loader", loaders)
def test_multiclass_arff(loader):
    """
    Test the multiclass arff datasets
    """

    dataset = loader()

    assert len(dataset) == 15

    assert dataset['data'].dtype == float

    assert dataset['target'].dtype == int

    assert dataset['data'].shape[0] > 0

    assert dataset['data'].shape[0] == dataset['target'].shape[0]

    assert dataset['n_col_non_unique_orig'] <= dataset['n_col']

    DecisionTreeClassifier().fit(dataset['data'], dataset['target'])
    assert True

def test_get_data_loaders():
    """
    Test the data loaders
    """

    assert len(mult_clas.get_data_loaders()) == \
                    len(mult_clas.get_filtered_data_loaders())

    assert len(mult_clas.get_filtered_data_loaders(sorting='n',
                                                    n_smallest=5)) == 5

    assert len(mult_clas.get_data_loaders('small')) > 0

    assert len(mult_clas.get_data_loaders('tiny')) > 0

    assert len(mult_clas.get_data_loaders('study')) > 0
