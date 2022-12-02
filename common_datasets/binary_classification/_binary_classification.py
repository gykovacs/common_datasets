"""
This module brings together all binary classification data loaders
"""

import json
import io
import pkgutil

import pandas as pd

from ._binary_classification_part0 import (load_abalone_17_vs_7_8_9_10,
load_abalone_19_vs_10_11_12_13,
load_abalone_20_vs_8_9_10,
load_abalone_21_vs_8,
load_abalone_3_vs_11,
load_abalone19,
load_abalone9_18,
load_ecoli_0_1_3_7_vs_2_6,
load_ecoli_0_1_4_6_vs_5,
load_ecoli_0_1_4_7_vs_2_3_5_6,
load_ecoli_0_1_4_7_vs_5_6,
load_ecoli_0_1_vs_2_3_5,
load_ecoli_0_1_vs_5,
load_ecoli_0_2_3_4_vs_5,
load_ecoli_0_2_6_7_vs_3_5,
load_ecoli_0_3_4_6_vs_5,
load_ecoli_0_3_4_7_vs_5_6,
load_ecoli_0_3_4_vs_5,
load_ecoli_0_4_6_vs_5,
load_ecoli_0_6_7_vs_3_5,
load_ecoli_0_6_7_vs_5,
load_ecoli4,
load_ecoli_0_vs_1,
load_ecoli1,
load_ecoli2,
load_ecoli3,
load_glass_0_1_4_6_vs_2,
load_glass_0_1_5_vs_2,
load_glass_0_1_6_vs_2,
load_glass_0_1_6_vs_5,
load_glass_0_4_vs_5,
load_glass_0_6_vs_5,
load_glass2,
load_glass4,
load_glass5,
load_glass_0_1_2_3_vs_4_5_6,
load_glass0,
load_glass1,
load_glass6,
load_yeast_0_2_5_6_vs_3_7_8_9,
load_yeast_0_2_5_7_9_vs_3_6_8,
load_yeast_0_3_5_9_vs_7_8,
load_yeast_0_5_6_7_9_vs_4,
load_yeast_1_2_8_9_vs_7,
load_yeast_1_4_5_8_vs_7,
load_yeast_1_vs_7,
load_yeast_2_vs_4,
load_yeast_2_vs_8,
load_yeast4,
load_yeast5,
load_yeast6,
load_yeast1,
load_yeast3,
load_winequality_red_3_vs_5,
load_winequality_red_4,
load_winequality_red_8_vs_6,
load_winequality_red_8_vs_6_7,
load_winequality_white_3_9_vs_5,
load_winequality_white_3_vs_7,
load_winequality_white_9_vs_4)

from ._binary_classification_part1 import (
load_kddcup_buffer_overflow_vs_back,
load_kddcup_guess_passwd_vs_satan,
load_kddcup_land_vs_portsweep,
load_kddcup_land_vs_satan,
load_kddcup_rootkit_imap_vs_back,
load_kr_vs_k_one_vs_fifteen,
load_kr_vs_k_three_vs_eleven,
load_kr_vs_k_zero_one_vs_draw,
load_kr_vs_k_zero_vs_eight,
load_kr_vs_k_zero_vs_fifteen,
load_poker_8_9_vs_5,
load_poker_8_9_vs_6,
load_poker_8_vs_6,
load_poker_9_vs_7,
load_shuttle_2_vs_5,
load_shuttle_6_vs_2_3,
load_shuttle_c0_vs_c4,
load_shuttle_c2_vs_c4,
load_vehicle0,
load_vehicle1,
load_vehicle2,
load_vehicle3,
load_cm1,
load_kc1,
load_pc1,
load_car_good,
load_car_vgood,
load_cleveland_0_vs_4,
load_dermatology_6,
load_flaref,
load_led7digit_0_2_4_5_6_7_8_9_vs_1,
load_lymphography_normal_fibrosis,
load_page_blocks_1_3_vs_4,
load_vowel0,
load_zoo_3,
load_haberman,
load_iris0,
load_new_thyroid1,
load_page_blocks0,
load_pima,
load_german,
load_hepatitis,
load_hypothyroid,
load_satimage,
load_spectf,
load_sylva,
load_segment0,
load_wisconsin,
load_mammographic,
load_bupa,
load_monk_2,
load_appendicitis,
load_saheart,
load_australian,
load_crx,
load_lymphography,
load_wdbc,
load_ionosphere,
load_ada
#load_hiva
)

summary = json.loads(pkgutil.get_data('common_datasets', 'data/summary_binary_classification.json').decode('utf-8'))

__all__= ['load_ada',
            #'load_hiva',
            'load_abalone_17_vs_7_8_9_10',
            'load_abalone_19_vs_10_11_12_13',
            'load_abalone_20_vs_8_9_10',
            'load_abalone_21_vs_8',
            'load_abalone_3_vs_11',
            'load_abalone19',
            'load_abalone9_18',
            'load_ecoli_0_1_3_7_vs_2_6',
            'load_ecoli_0_1_4_6_vs_5',
            'load_ecoli_0_1_4_7_vs_2_3_5_6',
            'load_ecoli_0_1_4_7_vs_5_6',
            'load_ecoli_0_1_vs_2_3_5',
            'load_ecoli_0_1_vs_5',
            'load_ecoli_0_2_3_4_vs_5',
            'load_ecoli_0_2_6_7_vs_3_5',
            'load_ecoli_0_3_4_6_vs_5',
            'load_ecoli_0_3_4_7_vs_5_6',
            'load_ecoli_0_3_4_vs_5',
            'load_ecoli_0_4_6_vs_5',
            'load_ecoli_0_6_7_vs_3_5',
            'load_ecoli_0_6_7_vs_5',
            'load_ecoli4',
            'load_ecoli_0_vs_1',
            'load_ecoli1',
            'load_ecoli2',
            'load_ecoli3',
            'load_glass_0_1_4_6_vs_2',
            'load_glass_0_1_5_vs_2',
            'load_glass_0_1_6_vs_2',
            'load_glass_0_1_6_vs_5',
            'load_glass_0_4_vs_5',
            'load_glass_0_6_vs_5',
            'load_glass2',
            'load_glass4',
            'load_glass5',
            'load_glass_0_1_2_3_vs_4_5_6',
            'load_glass0',
            'load_glass1',
            'load_glass6',
            'load_yeast_0_2_5_6_vs_3_7_8_9',
            'load_yeast_0_2_5_7_9_vs_3_6_8',
            'load_yeast_0_3_5_9_vs_7_8',
            'load_yeast_0_5_6_7_9_vs_4',
            'load_yeast_1_2_8_9_vs_7',
            'load_yeast_1_4_5_8_vs_7',
            'load_yeast_1_vs_7',
            'load_yeast_2_vs_4',
            'load_yeast_2_vs_8',
            'load_yeast4',
            'load_yeast5',
            'load_yeast6',
            'load_winequality_red_3_vs_5',
            'load_winequality_red_4',
            'load_winequality_red_8_vs_6',
            'load_winequality_red_8_vs_6_7',
            'load_winequality_white_3_9_vs_5',
            'load_winequality_white_3_vs_7',
            'load_winequality_white_9_vs_4',
            'load_kddcup_buffer_overflow_vs_back',
            'load_kddcup_guess_passwd_vs_satan',
            'load_kddcup_land_vs_portsweep',
            'load_kddcup_land_vs_satan',
            'load_kddcup_rootkit_imap_vs_back',
            'load_kr_vs_k_one_vs_fifteen',
            'load_kr_vs_k_three_vs_eleven',
            'load_kr_vs_k_zero_one_vs_draw',
            'load_kr_vs_k_zero_vs_eight',
            'load_kr_vs_k_zero_vs_fifteen',
            'load_poker_8_9_vs_5',
            'load_poker_8_9_vs_6',
            'load_poker_8_vs_6',
            'load_poker_9_vs_7',
            'load_shuttle_2_vs_5',
            'load_shuttle_6_vs_2_3',
            'load_shuttle_c0_vs_c4',
            'load_shuttle_c2_vs_c4',
            'load_vehicle0',
            'load_vehicle1',
            'load_vehicle2',
            'load_vehicle3',
            'load_cm1',
            'load_kc1',
            'load_pc1',
            'load_car_good',
            'load_car_vgood',
            'load_cleveland_0_vs_4',
            'load_dermatology_6',
            'load_flaref',
            'load_led7digit_0_2_4_5_6_7_8_9_vs_1',
            'load_lymphography_normal_fibrosis',
            'load_page_blocks_1_3_vs_4',
            'load_vowel0',
            'load_zoo_3',
            'load_haberman',
            'load_iris0',
            'load_new_thyroid1',
            'load_page_blocks0',
            'load_pima',
            'load_german',
            'load_hepatitis',
            'load_hypothyroid',
            'load_satimage',
            'load_spectf',
            'load_sylva',
            'load_segment0',
            'load_wisconsin',
            'load_yeast1',
            'load_yeast3',
            'load_mammographic',
            'load_bupa',
            'load_monk_2',
            'load_appendicitis',
            'load_saheart',
            'load_australian',
            'load_crx',
            'load_lymphography',
            'load_wdbc',
            'load_ionosphere',
            'get_data_loaders',
            'get_filtered_data_loaders',
            'summary_pdf']

summary_pdf = pd.DataFrame.from_dict(summary)

def get_filtered_data_loaders(*,
                              n_col_bounds=(1, 5000),
                              n_col_orig_bounds=(1, 5000),
                              n_bounds=(1, 10000),
                              n_minority_bounds=(1, 10000),
                              imbalance_ratio_bounds=(0, 1000),
                              n_smallest=-1,
                              sorting=None,
                              n_from_phenotypes=None):
    """
    Get filtered data loaders.

    Args:
        n_col_bounds (tuple): the lower and upper bounds on the number
                                of columns
        n_col_orig_bounds (tuple): the lower and upper bounds on the
                                    number of original columns
        n_bounds (tuple): the lower and upper bounds on the number
                            of records
        n_minority_bounds (tuple): the lower and upper bounds on the
                                    number of minority samples
        imbalance_ratio_bounds (tuple): the lower and upper bounds on
                                    the imbalance ratio
        n_smallest (int): the number of smallest in the sense of "sorting"
        sorting (str): the sorting attribute ('n', 'n_col', 'n_minority',
                            'imbalance_ratio')
        n_from_phenotypes (int): the maximum number of datasets from a
                                    phenotype

    Returns:
        list: the list of data loaders
    """

    descriptors= summary_pdf
    data_loaders = descriptors[(descriptors['n'] >= n_bounds[0])
                                & (descriptors['n'] < n_bounds[1])
                                & (descriptors['n_col'] >= n_col_bounds[0])
                                & (descriptors['n_col'] < n_col_bounds[1])
                                & (descriptors['n_col_orig'] >= n_col_orig_bounds[0])
                                & (descriptors['n_col_orig'] < n_col_orig_bounds[1])
                                & (descriptors['imbalance_ratio'] >= imbalance_ratio_bounds[0])
                                & (descriptors['imbalance_ratio'] < imbalance_ratio_bounds[1])
                                & (descriptors['n_minority'] >= n_minority_bounds[0])
                                & (descriptors['n_minority'] < n_minority_bounds[1])]

    if n_from_phenotypes is not None:
        data_loaders = data_loaders.groupby('phenotype').head(n_from_phenotypes)

    if sorting is not None:
        data_loaders = data_loaders.sort_values(sorting)

        if n_smallest != -1:
            data_loaders = data_loaders[:n_smallest]

    data_loaders = data_loaders['data_loader'].values

    return [globals()[data_loader] for data_loader in data_loaders]

def get_data_loaders(subset='all',
                    n_smallest=-1,
                    sorting=None,
                    n_from_phenotypes=None):
    """
    Get a subset of data loaders

    Args:
        subset (str): 'all'/'study'/'small'/'tiny'
        n_smallest (int): the number of smallest in the sense of "sorting"
        sorting (str): the sorting attribute ('n', 'n_col')
        n_from_phenotypes (int): the maximum number of datasets from a
                                    phenotype

    Returns:
        list: the list of data loaders
    """

    n_col_bounds = (1, 5000)
    n_col_orig_bounds = (1, 5000)
    n_bounds = (1, 10000)
    n_minority_bounds = (1, 10000)
    imbalance_ratio_bounds = (0, 1000)

    if subset == 'study':
        n_col_bounds = (n_col_bounds[0], 100)
        n_bounds = (n_bounds[0], 4000)
    elif subset == 'small':
        n_col_bounds = (n_col_bounds[0], 100)
        n_bounds = (n_bounds[0], 1000)
    elif subset == 'tiny':
        n_bounds = (n_bounds[0], 120)

    return get_filtered_data_loaders(n_col_bounds= n_col_bounds,
                                    n_col_orig_bounds= n_col_orig_bounds,
                                    n_bounds= n_bounds,
                                    n_minority_bounds= n_minority_bounds,
                                    imbalance_ratio_bounds= imbalance_ratio_bounds,
                                    n_smallest=n_smallest,
                                    sorting=sorting,
                                    n_from_phenotypes=n_from_phenotypes)
