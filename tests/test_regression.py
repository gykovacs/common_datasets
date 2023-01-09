"""
Testing regression datasets.
"""

import pytest

from sklearn.tree import DecisionTreeRegressor

from common_datasets import regression

loaders = [regression.load_diabetes,
           regression.load_laser,
           regression.load_autoMPG6,
           regression.load_wizmir,
           regression.load_wankara,
           regression.load_mortgage,
           regression.load_baseball,
           regression.load_compactiv,
           regression.load_treasury,
           regression.load_puma32h,
           regression.load_airfoil,
           regression.load_cpu_performance,
           regression.load_forestfires,
           regression.load_slump_test,
           regression.load_winequality_red,
           regression.load_winequality_white,
           regression.load_yacht_hydrodynamics,
           regression.load_communities,
           regression.load_real_estate_valuation,
           regression.load_stock_portfolio_performance,
           regression.load_residential_building,
           regression.load_ccpp,
           regression.load_o_ring,
           regression.load_daily_demand,
           regression.load_wsn_ale,
           regression.load_servo,
           regression.load_qsar_aquatic_toxicity,
           regression.load_excitation_current,
           regression.load_qsar_fish_toxicity]

@pytest.mark.parametrize("loader", loaders)
def test_regression(loader):
    """
    Test the regression datasets
    """

    dataset = loader()

    assert len(dataset) == 17

    assert dataset['data'].dtype == float

    assert dataset['target'].dtype == float

    assert dataset['data'].shape[0] > 0

    assert dataset['data'].shape[0] == dataset['target'].shape[0]

    assert dataset['n_col_non_unique_orig'] <= dataset['n_col']

    DecisionTreeRegressor().fit(dataset['data'], dataset['target'])
    assert True

def test_get_data_loaders():
    """
    Test the data loaders
    """

    assert len(regression.get_data_loaders()) == \
                    len(regression.get_filtered_data_loaders())

    assert len(regression.get_filtered_data_loaders(sorting='n',
                                                    n_smallest=5)) == 5

    assert len(regression.get_data_loaders('small')) > 0

    assert len(regression.get_data_loaders('tiny')) > 0

    assert len(regression.get_data_loaders('study')) > 0

    # this should change if more datasets from the same phenotype are added
    assert len(regression.get_data_loaders()) >= \
                len(regression.get_data_loaders(n_from_phenotypes=1))

    assert len(regression.get_summary_pdf()) > 0
