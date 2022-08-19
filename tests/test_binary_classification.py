"""
Testing the binary classification datasets.
"""

import pytest

import numpy as np

from sklearn.tree import DecisionTreeClassifier

import common_datasets.binary_classification as bin_clas

loaders = [bin_clas.load_abalone_17_vs_7_8_9_10,
            bin_clas.load_abalone_19_vs_10_11_12_13,
            bin_clas.load_abalone_20_vs_8_9_10,
            bin_clas.load_abalone_21_vs_8,
            bin_clas.load_abalone_3_vs_11,
            bin_clas.load_abalone19,
            bin_clas.load_abalone9_18,
            bin_clas.load_ecoli_0_1_3_7_vs_2_6,
            bin_clas.load_ecoli_0_1_4_6_vs_5,
            bin_clas.load_ecoli_0_1_4_7_vs_2_3_5_6,
            bin_clas.load_ecoli_0_1_4_7_vs_5_6,
            bin_clas.load_ecoli_0_1_vs_2_3_5,
            bin_clas.load_ecoli_0_1_vs_5,
            bin_clas.load_ecoli_0_2_3_4_vs_5,
            bin_clas.load_ecoli_0_2_6_7_vs_3_5,
            bin_clas.load_ecoli_0_3_4_6_vs_5,
            bin_clas.load_ecoli_0_3_4_7_vs_5_6,
            bin_clas.load_ecoli_0_3_4_vs_5,
            bin_clas.load_ecoli_0_4_6_vs_5,
            bin_clas.load_ecoli_0_6_7_vs_3_5,
            bin_clas.load_ecoli_0_6_7_vs_5 ,
            bin_clas.load_ecoli4,
            bin_clas.load_ecoli_0_vs_1,
            bin_clas.load_ecoli1,
            bin_clas.load_ecoli2,
            bin_clas.load_ecoli3,
            bin_clas.load_glass_0_1_4_6_vs_2,
            bin_clas.load_glass_0_1_5_vs_2,
            bin_clas.load_glass_0_1_6_vs_2,
            bin_clas.load_glass_0_1_6_vs_5,
            bin_clas.load_glass_0_4_vs_5,
            bin_clas.load_glass_0_6_vs_5,
            bin_clas.load_glass2,
            bin_clas.load_glass4,
            bin_clas.load_glass5,
            bin_clas.load_glass_0_1_2_3_vs_4_5_6,
            bin_clas.load_glass0,
            bin_clas.load_glass1,
            bin_clas.load_glass6,
            bin_clas.load_kddcup_buffer_overflow_vs_back,
            bin_clas.load_kddcup_guess_passwd_vs_satan,
            bin_clas.load_kddcup_land_vs_portsweep,
            bin_clas.load_kddcup_land_vs_satan,
            bin_clas.load_kddcup_rootkit_imap_vs_back,
            bin_clas.load_kr_vs_k_one_vs_fifteen,
            bin_clas.load_kr_vs_k_three_vs_eleven,
            bin_clas.load_kr_vs_k_zero_one_vs_draw,
            bin_clas.load_kr_vs_k_zero_vs_eight,
            bin_clas.load_kr_vs_k_zero_vs_fifteen,
            bin_clas.load_cm1,
            bin_clas.load_kc1,
            bin_clas.load_pc1,
            bin_clas.load_car_good,
            bin_clas.load_car_vgood,
            bin_clas.load_cleveland_0_vs_4,
            bin_clas.load_dermatology_6,
            bin_clas.load_flaref,
            bin_clas.load_led7digit_0_2_4_5_6_7_8_9_vs_1,
            bin_clas.load_lymphography_normal_fibrosis,
            bin_clas.load_page_blocks_1_3_vs_4,
            bin_clas.load_vowel0,
            bin_clas.load_zoo_3,
            bin_clas.load_haberman,
            bin_clas.load_iris0,
            bin_clas.load_new_thyroid1,
            bin_clas.load_page_blocks0,
            bin_clas.load_pima,
            bin_clas.load_segment0,
            bin_clas.load_wisconsin,
            bin_clas.load_mammographic,
            bin_clas.load_monk_2,
            bin_clas.load_appendicitis,
            bin_clas.load_saheart,
            bin_clas.load_australian,
            bin_clas.load_wdbc,
            bin_clas.load_ionosphere,
            bin_clas.load_spectfheart,
            bin_clas.load_bupa,
            bin_clas.load_crx,
            bin_clas.load_lymphography,
            bin_clas.load_poker_8_9_vs_5,
            bin_clas.load_poker_8_9_vs_6,
            bin_clas.load_poker_8_vs_6,
            bin_clas.load_poker_9_vs_7,
            bin_clas.load_shuttle_2_vs_5,
            bin_clas.load_shuttle_6_vs_2_3,
            bin_clas.load_shuttle_c0_vs_c4,
            bin_clas.load_shuttle_c2_vs_c4,
            bin_clas.load_vehicle0,
            bin_clas.load_vehicle1,
            bin_clas.load_vehicle2,
            bin_clas.load_vehicle3,
            bin_clas.load_winequality_red_3_vs_5,
            bin_clas.load_winequality_red_4,
            bin_clas.load_winequality_red_8_vs_6,
            bin_clas.load_winequality_red_8_vs_6_7,
            bin_clas.load_winequality_white_3_9_vs_5,
            bin_clas.load_winequality_white_3_vs_7,
            bin_clas.load_winequality_white_9_vs_4,
            bin_clas.load_yeast_0_2_5_6_vs_3_7_8_9,
            bin_clas.load_yeast_0_2_5_7_9_vs_3_6_8,
            bin_clas.load_yeast_0_3_5_9_vs_7_8,
            bin_clas.load_yeast_0_5_6_7_9_vs_4,
            bin_clas.load_yeast_1_2_8_9_vs_7,
            bin_clas.load_yeast_1_4_5_8_vs_7,
            bin_clas.load_yeast_1_vs_7,
            bin_clas.load_yeast_2_vs_4,
            bin_clas.load_yeast_2_vs_8,
            bin_clas.load_yeast4,
            bin_clas.load_yeast5,
            bin_clas.load_yeast6,
            bin_clas.load_yeast1,
            bin_clas.load_yeast3,
            bin_clas.load_ada,
            bin_clas.load_hiva,
            bin_clas.load_hypothyroid,
            bin_clas.load_sylva,
            bin_clas.load_spectf,
            bin_clas.load_hepatitis,
            bin_clas.load_german,
            bin_clas.load_satimage]

@pytest.mark.parametrize("loader", loaders)
def test_binary(loader):
    """
    Test the binary classification datasets
    """

    dataset = loader()

    assert len(dataset) == 16

    assert dataset['data'].dtype == float

    assert dataset['target'].dtype == int

    assert dataset['data'].shape[0] > 0

    assert dataset['data'].shape[0] == dataset['target'].shape[0]

    assert np.sum(dataset['target'] == 0) >= np.sum(dataset['target'] == 1)

    assert dataset['n_col_non_unique_orig'] <= dataset['n_col']

    DecisionTreeClassifier().fit(dataset['data'], dataset['target'])
    assert True

def test_get_data_loaders():
    """
    Test the data loaders
    """

    assert len(bin_clas.get_data_loaders()) == \
                    len(bin_clas.get_filtered_data_loaders())

    assert len(bin_clas.get_filtered_data_loaders(sorting='n',
                                                    n_smallest=5)) == 5

    assert len(bin_clas.get_data_loaders('small')) > 0

    assert len(bin_clas.get_data_loaders('tiny')) > 0

    assert len(bin_clas.get_data_loaders('study')) > 0
