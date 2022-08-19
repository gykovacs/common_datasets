"""
This module contains the summary of all binary classification problems, part 0
"""

summary_part0 = [{'name': 'abalone19',
  'phenotype': 'abalone',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 8,
  'n_col_non_unique_orig': 8,
  'n': 4174,
  'DESCR': 'abalone19',
  'n_minority': 32,
  'imbalance_ratio': 129.4375,
  'data_loader': 'load_abalone19'},
 {'name': 'abalone9_18',
  'phenotype': 'abalone',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 8,
  'n_col_non_unique_orig': 8,
  'n': 731,
  'DESCR': 'abalone9_18',
  'n_minority': 42,
  'imbalance_ratio': 16.404761904761905,
  'data_loader': 'load_abalone9_18'},
 {'name': 'abalone-17_vs_7-8-9-10',
  'phenotype': 'abalone',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 8,
  'n_col_non_unique_orig': 8,
  'n': 2338,
  'DESCR': 'abalone-17_vs_7-8-9-10',
  'n_minority': 58,
  'imbalance_ratio': 39.310344827586206,
  'data_loader': 'load_abalone_17_vs_7_8_9_10'},
 {'name': 'abalone-19_vs_10-11-12-13',
  'phenotype': 'abalone',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 8,
  'n_col_non_unique_orig': 8,
  'n': 1622,
  'DESCR': 'abalone-19_vs_10-11-12-13',
  'n_minority': 32,
  'imbalance_ratio': 49.6875,
  'data_loader': 'load_abalone_19_vs_10_11_12_13'},
 {'name': 'abalone-20_vs_8_9_10',
  'phenotype': 'abalone',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 8,
  'n_col_non_unique_orig': 8,
  'n': 1916,
  'DESCR': 'abalone-20_vs_8_9_10',
  'n_minority': 26,
  'imbalance_ratio': 72.6923076923077,
  'data_loader': 'load_abalone_20_vs_8_9_10'},
 {'name': 'abalone-22_vs_8',
  'phenotype': 'abalone',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 8,
  'n_col_non_unique_orig': 8,
  'n': 581,
  'DESCR': 'abalone-22_vs_8',
  'n_minority': 14,
  'imbalance_ratio': 40.5,
  'data_loader': 'load_abalone_21_vs_8'},
 {'name': 'abalone-3_vs_11',
  'phenotype': 'abalone',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 8,
  'n_col_non_unique_orig': 8,
  'n': 502,
  'DESCR': 'abalone-3_vs_11',
  'n_minority': 15,
  'imbalance_ratio': 32.46666666666667,
  'data_loader': 'load_abalone_3_vs_11'},
 {'name': 'ADA',
  'phenotype': 'ADA',
  'citation_key': 'krnn',
  'n_col': 48,
  'n_col_orig': 48,
  'n_col_non_unique_orig': 47,
  'n': 4147,
  'DESCR': 'ADA',
  'n_minority': 1029,
  'imbalance_ratio': 3.0301263362487854,
  'data_loader': 'load_ada'},
 {'name': 'appendicitis',
  'phenotype': 'appendicitis',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 106,
  'DESCR': 'appendicitis',
  'n_minority': 21,
  'imbalance_ratio': 4.0476190476190474,
  'data_loader': 'load_appendicitis'},
 {'name': 'australian',
  'phenotype': 'australian',
  'citation_key': 'keel',
  'n_col': 16,
  'n_col_orig': 14,
  'n_col_non_unique_orig': 14,
  'n': 690,
  'DESCR': 'australian',
  'n_minority': 307,
  'imbalance_ratio': 1.247557003257329,
  'data_loader': 'load_australian'},
 {'name': 'bupa',
  'phenotype': 'bupa',
  'citation_key': 'keel',
  'n_col': 6,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 345,
  'DESCR': 'bupa',
  'n_minority': 200,
  'imbalance_ratio': 0.725,
  'data_loader': 'load_bupa'},
 {'name': 'car_good',
  'phenotype': 'car',
  'citation_key': 'keel',
  'n_col': 15,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 1728,
  'DESCR': 'car_good',
  'n_minority': 69,
  'imbalance_ratio': 24.043478260869566,
  'data_loader': 'load_car_good'},
 {'name': 'car-vgood',
  'phenotype': 'car',
  'citation_key': 'keel',
  'n_col': 15,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 1728,
  'DESCR': 'car-vgood',
  'n_minority': 65,
  'imbalance_ratio': 25.584615384615386,
  'data_loader': 'load_car_vgood'},
 {'name': 'cleveland-0_vs_4',
  'phenotype': 'cleveland',
  'citation_key': 'keel',
  'n_col': 13,
  'n_col_orig': 13,
  'n_col_non_unique_orig': 13,
  'n': 177,
  'DESCR': 'cleveland-0_vs_4',
  'n_minority': 13,
  'imbalance_ratio': 12.615384615384615,
  'data_loader': 'load_cleveland_0_vs_4'},
 {'name': 'CM1',
  'phenotype': 'CM',
  'citation_key': 'krnn',
  'n_col': 21,
  'n_col_orig': 21,
  'n_col_non_unique_orig': 21,
  'n': 498,
  'DESCR': 'CM1',
  'n_minority': 49,
  'imbalance_ratio': 9.16326530612245,
  'data_loader': 'load_cm1'},
 {'name': 'crx',
  'phenotype': 'crx',
  'citation_key': 'keel',
  'n_col': 37,
  'n_col_orig': 15,
  'n_col_non_unique_orig': 15,
  'n': 653,
  'DESCR': 'crx',
  'n_minority': 296,
  'imbalance_ratio': 1.2060810810810811,
  'data_loader': 'load_crx'},
 {'name': 'dermatology-6',
  'phenotype': 'dermatology',
  'citation_key': 'keel',
  'n_col': 34,
  'n_col_orig': 34,
  'n_col_non_unique_orig': 34,
  'n': 358,
  'DESCR': 'dermatology-6',
  'n_minority': 20,
  'imbalance_ratio': 16.9,
  'data_loader': 'load_dermatology_6'},
 {'name': 'ecoli1',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 336,
  'DESCR': 'ecoli1',
  'n_minority': 77,
  'imbalance_ratio': 3.3636363636363638,
  'data_loader': 'load_ecoli1'},
 {'name': 'ecoli2',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 336,
  'DESCR': 'ecoli2',
  'n_minority': 52,
  'imbalance_ratio': 5.461538461538462,
  'data_loader': 'load_ecoli2'},
 {'name': 'ecoli3',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 336,
  'DESCR': 'ecoli3',
  'n_minority': 35,
  'imbalance_ratio': 8.6,
  'data_loader': 'load_ecoli3'},
 {'name': 'ecoli4',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 336,
  'DESCR': 'ecoli4',
  'n_minority': 20,
  'imbalance_ratio': 15.8,
  'data_loader': 'load_ecoli4'},
 {'name': 'ecoli_0_1_3_7_vs_2_6',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 281,
  'DESCR': 'ecoli_0_1_3_7_vs_2_6',
  'n_minority': 7,
  'imbalance_ratio': 39.142857142857146,
  'data_loader': 'load_ecoli_0_1_3_7_vs_2_6'},
 {'name': 'ecoli_0_1_4_6_vs_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 6,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 280,
  'DESCR': 'ecoli_0_1_4_6_vs_5',
  'n_minority': 20,
  'imbalance_ratio': 13.0,
  'data_loader': 'load_ecoli_0_1_4_6_vs_5'},
 {'name': 'ecoli_0_1_4_7_vs_2_3_5_6',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 336,
  'DESCR': 'ecoli_0_1_4_7_vs_2_3_5_6',
  'n_minority': 29,
  'imbalance_ratio': 10.586206896551724,
  'data_loader': 'load_ecoli_0_1_4_7_vs_2_3_5_6'},
 {'name': 'ecoli_0_1_4_7_vs_5_6',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 6,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 332,
  'DESCR': 'ecoli_0_1_4_7_vs_5_6',
  'n_minority': 25,
  'imbalance_ratio': 12.28,
  'data_loader': 'load_ecoli_0_1_4_7_vs_5_6'},
 {'name': 'ecoli_0_1_vs_2_3_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 244,
  'DESCR': 'ecoli_0_1_vs_2_3_5',
  'n_minority': 24,
  'imbalance_ratio': 9.166666666666666,
  'data_loader': 'load_ecoli_0_1_vs_2_3_5'},
 {'name': 'ecoli_0_1_vs_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 6,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 240,
  'DESCR': 'ecoli_0_1_vs_5',
  'n_minority': 20,
  'imbalance_ratio': 11.0,
  'data_loader': 'load_ecoli_0_1_vs_5'},
 {'name': 'ecoli_0_2_3_4_vs_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 202,
  'DESCR': 'ecoli_0_2_3_4_vs_5',
  'n_minority': 20,
  'imbalance_ratio': 9.1,
  'data_loader': 'load_ecoli_0_2_3_4_vs_5'},
 {'name': 'ecoli_0_2_6_7_vs_3_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 224,
  'DESCR': 'ecoli_0_2_6_7_vs_3_5',
  'n_minority': 22,
  'imbalance_ratio': 9.181818181818182,
  'data_loader': 'load_ecoli_0_2_6_7_vs_3_5'},
 {'name': 'ecoli_0_3_4_6_vs_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 205,
  'DESCR': 'ecoli_0_3_4_6_vs_5',
  'n_minority': 20,
  'imbalance_ratio': 9.25,
  'data_loader': 'load_ecoli_0_3_4_6_vs_5'},
 {'name': 'ecoli_0_3_4_7_vs_5_6',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 257,
  'DESCR': 'ecoli_0_3_4_7_vs_5_6',
  'n_minority': 25,
  'imbalance_ratio': 9.28,
  'data_loader': 'load_ecoli_0_3_4_7_vs_5_6'},
 {'name': 'ecoli_0_3_4_vs_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 200,
  'DESCR': 'ecoli_0_3_4_vs_5',
  'n_minority': 20,
  'imbalance_ratio': 9.0,
  'data_loader': 'load_ecoli_0_3_4_vs_5'},
 {'name': 'ecoli_0_4_6_vs_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 6,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 203,
  'DESCR': 'ecoli_0_4_6_vs_5',
  'n_minority': 20,
  'imbalance_ratio': 9.15,
  'data_loader': 'load_ecoli_0_4_6_vs_5'},
 {'name': 'ecoli_0_6_7_vs_3_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 222,
  'DESCR': 'ecoli_0_6_7_vs_3_5',
  'n_minority': 22,
  'imbalance_ratio': 9.090909090909092,
  'data_loader': 'load_ecoli_0_6_7_vs_3_5'},
 {'name': 'ecoli_0_6_7_vs_5',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 6,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 220,
  'DESCR': 'ecoli_0_6_7_vs_5',
  'n_minority': 20,
  'imbalance_ratio': 10.0,
  'data_loader': 'load_ecoli_0_6_7_vs_5'},
 {'name': 'ecoli_0_vs_1',
  'phenotype': 'ecoli',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 6,
  'n': 220,
  'DESCR': 'ecoli_0_vs_1',
  'n_minority': 143,
  'imbalance_ratio': 0.5384615384615384,
  'data_loader': 'load_ecoli_0_vs_1'},
 {'name': 'flare-F',
  'phenotype': 'flare',
  'citation_key': 'keel',
  'n_col': 30,
  'n_col_orig': 11,
  'n_col_non_unique_orig': 11,
  'n': 1066,
  'DESCR': 'flare-F',
  'n_minority': 43,
  'imbalance_ratio': 23.790697674418606,
  'data_loader': 'load_flaref'},
 {'name': 'german',
  'phenotype': 'german',
  'citation_key': 'krnn',
  'n_col': 998,
  'n_col_orig': 0,
  'n_col_non_unique_orig': 0,
  'n': 1000,
  'DESCR': 'german',
  'n_minority': 1,
  'imbalance_ratio': 999.0,
  'data_loader': 'load_german'},
 {'name': 'glass0',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 214,
  'DESCR': 'glass0',
  'n_minority': 70,
  'imbalance_ratio': 2.057142857142857,
  'data_loader': 'load_glass0'},
 {'name': 'glass1',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 214,
  'DESCR': 'glass1',
  'n_minority': 76,
  'imbalance_ratio': 1.8157894736842106,
  'data_loader': 'load_glass1'},
 {'name': 'glass2',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 214,
  'DESCR': 'glass2',
  'n_minority': 17,
  'imbalance_ratio': 11.588235294117647,
  'data_loader': 'load_glass2'},
 {'name': 'glass4',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 214,
  'DESCR': 'glass4',
  'n_minority': 13,
  'imbalance_ratio': 15.461538461538462,
  'data_loader': 'load_glass4'},
 {'name': 'glass5',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 214,
  'DESCR': 'glass5',
  'n_minority': 9,
  'imbalance_ratio': 22.77777777777778,
  'data_loader': 'load_glass5'},
 {'name': 'glass6',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 214,
  'DESCR': 'glass6',
  'n_minority': 29,
  'imbalance_ratio': 6.379310344827586,
  'data_loader': 'load_glass6'},
 {'name': 'glass_0_1_2_3_vs_4_5_6',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 214,
  'DESCR': 'glass_0_1_2_3_vs_4_5_6',
  'n_minority': 51,
  'imbalance_ratio': 3.196078431372549,
  'data_loader': 'load_glass_0_1_2_3_vs_4_5_6'},
 {'name': 'glass_0_1_4_6_vs_2',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 205,
  'DESCR': 'glass_0_1_4_6_vs_2',
  'n_minority': 17,
  'imbalance_ratio': 11.058823529411764,
  'data_loader': 'load_glass_0_1_4_6_vs_2'},
 {'name': 'glass_0_1_5_vs_2',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 172,
  'DESCR': 'glass_0_1_5_vs_2',
  'n_minority': 17,
  'imbalance_ratio': 9.117647058823529,
  'data_loader': 'load_glass_0_1_5_vs_2'},
 {'name': 'glass_0_1_6_vs_2',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 192,
  'DESCR': 'glass_0_1_6_vs_2',
  'n_minority': 17,
  'imbalance_ratio': 10.294117647058824,
  'data_loader': 'load_glass_0_1_6_vs_2'},
 {'name': 'glass_0_1_6_vs_5',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 184,
  'DESCR': 'glass_0_1_6_vs_5',
  'n_minority': 9,
  'imbalance_ratio': 19.444444444444443,
  'data_loader': 'load_glass_0_1_6_vs_5'},
 {'name': 'glass_0_4_vs_5',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 92,
  'DESCR': 'glass_0_4_vs_5',
  'n_minority': 9,
  'imbalance_ratio': 9.222222222222221,
  'data_loader': 'load_glass_0_4_vs_5'},
 {'name': 'glass_0_6_vs_5',
  'phenotype': 'glass',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 108,
  'DESCR': 'glass_0_6_vs_5',
  'n_minority': 9,
  'imbalance_ratio': 11.0,
  'data_loader': 'load_glass_0_6_vs_5'},
 {'name': 'haberman',
  'phenotype': 'haberman',
  'citation_key': 'keel',
  'n_col': 3,
  'n_col_orig': 3,
  'n_col_non_unique_orig': 3,
  'n': 306,
  'DESCR': 'haberman',
  'n_minority': 81,
  'imbalance_ratio': 2.7777777777777777,
  'data_loader': 'load_haberman'},
 {'name': 'hepatitis',
  'phenotype': 'hepatitis',
  'citation_key': 'krnn',
  'n_col': 34,
  'n_col_orig': 19,
  'n_col_non_unique_orig': 19,
  'n': 155,
  'DESCR': 'hepatitis',
  'n_minority': 123,
  'imbalance_ratio': 0.2601626016260163,
  'data_loader': 'load_hepatitis'},
 {'name': 'HIVA',
  'phenotype': 'HIVA',
  'citation_key': 'krnn',
  'n_col': 1617,
  'n_col_orig': 1617,
  'n_col_non_unique_orig': 1616,
  'n': 3845,
  'DESCR': 'HIVA',
  'n_minority': 135,
  'imbalance_ratio': 27.48148148148148,
  'data_loader': 'load_hiva'},
 {'name': 'hypothyroid',
  'phenotype': 'hypothyroid',
  'citation_key': 'krnn',
  'n_col': 33,
  'n_col_orig': 25,
  'n_col_non_unique_orig': 24,
  'n': 3163,
  'DESCR': 'hypothyroid',
  'n_minority': 3012,
  'imbalance_ratio': 0.050132802124833994,
  'data_loader': 'load_hypothyroid'},
 {'name': 'ionosphere',
  'phenotype': 'ionosphere',
  'citation_key': 'keel',
  'n_col': 33,
  'n_col_orig': 33,
  'n_col_non_unique_orig': 33,
  'n': 351,
  'DESCR': 'ionosphere',
  'n_minority': 225,
  'imbalance_ratio': 0.56,
  'data_loader': 'load_ionosphere'},
 {'name': 'iris0',
  'phenotype': 'iris',
  'citation_key': 'keel',
  'n_col': 4,
  'n_col_orig': 4,
  'n_col_non_unique_orig': 4,
  'n': 150,
  'DESCR': 'iris0',
  'n_minority': 50,
  'imbalance_ratio': 2.0,
  'data_loader': 'load_iris0'},
 {'name': 'KC1',
  'phenotype': 'KC',
  'citation_key': 'krnn',
  'n_col': 21,
  'n_col_orig': 21,
  'n_col_non_unique_orig': 21,
  'n': 2109,
  'DESCR': 'KC1',
  'n_minority': 326,
  'imbalance_ratio': 5.469325153374233,
  'data_loader': 'load_kc1'},
 {'name': 'kddcup-buffer_overflow_vs_back',
  'phenotype': 'kddcup',
  'citation_key': 'keel',
  'n_col': 34,
  'n_col_orig': 41,
  'n_col_non_unique_orig': 27,
  'n': 2233,
  'DESCR': 'kddcup-buffer_overflow_vs_back',
  'n_minority': 30,
  'imbalance_ratio': 73.43333333333334,
  'data_loader': 'load_kddcup_buffer_overflow_vs_back'},
 {'name': 'kddcup-guess_passwd_vs_satan',
  'phenotype': 'kddcup',
  'citation_key': 'keel',
  'n_col': 60,
  'n_col_orig': 41,
  'n_col_non_unique_orig': 29,
  'n': 1642,
  'DESCR': 'kddcup-guess_passwd_vs_satan',
  'n_minority': 53,
  'imbalance_ratio': 29.9811320754717,
  'data_loader': 'load_kddcup_guess_passwd_vs_satan'},
 {'name': 'kddcup-land_vs_portsweep',
  'phenotype': 'kddcup',
  'citation_key': 'keel',
  'n_col': 68,
  'n_col_orig': 41,
  'n_col_non_unique_orig': 26,
  'n': 1061,
  'DESCR': 'kddcup-land_vs_portsweep',
  'n_minority': 21,
  'imbalance_ratio': 49.523809523809526,
  'data_loader': 'load_kddcup_land_vs_portsweep'},
 {'name': 'kddcup-land_vs_satan',
  'phenotype': 'kddcup',
  'citation_key': 'keel',
  'n_col': 55,
  'n_col_orig': 41,
  'n_col_non_unique_orig': 28,
  'n': 1610,
  'DESCR': 'kddcup-land_vs_satan',
  'n_minority': 21,
  'imbalance_ratio': 75.66666666666667,
  'data_loader': 'load_kddcup_land_vs_satan'},
 {'name': 'kddcup-rootkit-imap_vs_back',
  'phenotype': 'kddcup',
  'citation_key': 'keel',
  'n_col': 41,
  'n_col_orig': 41,
  'n_col_non_unique_orig': 33,
  'n': 2225,
  'DESCR': 'kddcup-rootkit-imap_vs_back',
  'n_minority': 22,
  'imbalance_ratio': 100.13636363636364,
  'data_loader': 'load_kddcup_rootkit_imap_vs_back'},
 {'name': 'kr_vs_k_one_vs_fifteen',
  'phenotype': 'kr',
  'citation_key': 'keel',
  'n_col': 33,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 2244,
  'DESCR': 'kr_vs_k_one_vs_fifteen',
  'n_minority': 78,
  'imbalance_ratio': 27.76923076923077,
  'data_loader': 'load_kr_vs_k_one_vs_fifteen'},
 {'name': 'kr-vs-k-three_vs_eleven',
  'phenotype': 'kr',
  'citation_key': 'keel',
  'n_col': 34,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 2935,
  'DESCR': 'kr-vs-k-three_vs_eleven',
  'n_minority': 81,
  'imbalance_ratio': 35.23456790123457,
  'data_loader': 'load_kr_vs_k_three_vs_eleven'},
 {'name': 'kr-vs-k-zero-one_vs_draw',
  'phenotype': 'kr',
  'citation_key': 'keel',
  'n_col': 34,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 2901,
  'DESCR': 'kr-vs-k-zero-one_vs_draw',
  'n_minority': 105,
  'imbalance_ratio': 26.62857142857143,
  'data_loader': 'load_kr_vs_k_zero_one_vs_draw'},
 {'name': 'kr-vs-k-zero_vs_eight',
  'phenotype': 'kr',
  'citation_key': 'keel',
  'n_col': 34,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 1460,
  'DESCR': 'kr-vs-k-zero_vs_eight',
  'n_minority': 27,
  'imbalance_ratio': 53.074074074074076,
  'data_loader': 'load_kr_vs_k_zero_vs_eight'},
 {'name': 'kr-vs-k-zero_vs_fifteen',
  'phenotype': 'kr',
  'citation_key': 'keel',
  'n_col': 33,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 2193,
  'DESCR': 'kr-vs-k-zero_vs_fifteen',
  'n_minority': 27,
  'imbalance_ratio': 80.22222222222223,
  'data_loader': 'load_kr_vs_k_zero_vs_fifteen'},
 {'name': 'led7digit-0-2-4-6-7-8-9_vs_1',
  'phenotype': 'led7digit',
  'citation_key': 'keel',
  'n_col': 7,
  'n_col_orig': 7,
  'n_col_non_unique_orig': 7,
  'n': 443,
  'DESCR': 'led7digit-0-2-4-6-7-8-9_vs_1',
  'n_minority': 37,
  'imbalance_ratio': 10.972972972972974,
  'data_loader': 'load_led7digit_0_2_4_5_6_7_8_9_vs_1'},
 {'name': 'lymphography',
  'phenotype': 'lymphography',
  'citation_key': 'keel',
  'n_col': 32,
  'n_col_orig': 18,
  'n_col_non_unique_orig': 18,
  'n': 148,
  'DESCR': 'lymphography',
  'n_minority': 4,
  'imbalance_ratio': 36.0,
  'data_loader': 'load_lymphography'},
 {'name': 'lymphography-normal-fibrosis',
  'phenotype': 'lymphography',
  'citation_key': 'keel',
  'n_col': 32,
  'n_col_orig': 18,
  'n_col_non_unique_orig': 18,
  'n': 148,
  'DESCR': 'lymphography-normal-fibrosis',
  'n_minority': 6,
  'imbalance_ratio': 23.666666666666668,
  'data_loader': 'load_lymphography_normal_fibrosis'},
 {'name': 'mammographic',
  'phenotype': 'mammographic',
  'citation_key': 'keel',
  'n_col': 5,
  'n_col_orig': 5,
  'n_col_non_unique_orig': 5,
  'n': 830,
  'DESCR': 'mammographic',
  'n_minority': 403,
  'imbalance_ratio': 1.0595533498759304,
  'data_loader': 'load_mammographic'},
 {'name': 'australian',
  'phenotype': 'australian',
  'citation_key': 'keel',
  'n_col': 6,
  'n_col_orig': 6,
  'n_col_non_unique_orig': 6,
  'n': 432,
  'DESCR': 'australian',
  'n_minority': 228,
  'imbalance_ratio': 0.8947368421052632,
  'data_loader': 'load_monk_2'},
 {'name': 'new_thyroid1',
  'phenotype': 'new',
  'citation_key': 'keel',
  'n_col': 5,
  'n_col_orig': 5,
  'n_col_non_unique_orig': 5,
  'n': 215,
  'DESCR': 'new_thyroid1',
  'n_minority': 35,
  'imbalance_ratio': 5.142857142857143,
  'data_loader': 'load_new_thyroid1'},
 {'name': 'new_thyroid2',
  'phenotype': 'new',
  'citation_key': 'keel',
  'n_col': 10,
  'n_col_orig': 10,
  'n_col_non_unique_orig': 10,
  'n': 5472,
  'DESCR': 'new_thyroid2',
  'n_minority': 559,
  'imbalance_ratio': 8.788908765652952,
  'data_loader': 'load_page_blocks0'},
 {'name': 'page-blocks-1-3_vs_4',
  'phenotype': 'page',
  'citation_key': 'keel',
  'n_col': 10,
  'n_col_orig': 10,
  'n_col_non_unique_orig': 10,
  'n': 472,
  'DESCR': 'page-blocks-1-3_vs_4',
  'n_minority': 28,
  'imbalance_ratio': 15.857142857142858,
  'data_loader': 'load_page_blocks_1_3_vs_4'},
 {'name': 'PC1',
  'phenotype': 'PC',
  'citation_key': 'krnn',
  'n_col': 21,
  'n_col_orig': 21,
  'n_col_non_unique_orig': 21,
  'n': 1109,
  'DESCR': 'PC1',
  'n_minority': 77,
  'imbalance_ratio': 13.402597402597403,
  'data_loader': 'load_pc1'},
 {'name': 'pima',
  'phenotype': 'pima',
  'citation_key': 'keel',
  'n_col': 8,
  'n_col_orig': 8,
  'n_col_non_unique_orig': 8,
  'n': 768,
  'DESCR': 'pima',
  'n_minority': 268,
  'imbalance_ratio': 1.8656716417910448,
  'data_loader': 'load_pima'},
 {'name': 'poker-8-9_vs_5',
  'phenotype': 'poker',
  'citation_key': 'keel',
  'n_col': 10,
  'n_col_orig': 10,
  'n_col_non_unique_orig': 10,
  'n': 2075,
  'DESCR': 'poker-8-9_vs_5',
  'n_minority': 25,
  'imbalance_ratio': 82.0,
  'data_loader': 'load_poker_8_9_vs_5'},
 {'name': 'poker-8-9_vs_6',
  'phenotype': 'poker',
  'citation_key': 'keel',
  'n_col': 10,
  'n_col_orig': 10,
  'n_col_non_unique_orig': 10,
  'n': 1485,
  'DESCR': 'poker-8-9_vs_6',
  'n_minority': 25,
  'imbalance_ratio': 58.4,
  'data_loader': 'load_poker_8_9_vs_6'},
 {'name': 'poker-8_vs_6',
  'phenotype': 'poker',
  'citation_key': 'keel',
  'n_col': 10,
  'n_col_orig': 10,
  'n_col_non_unique_orig': 10,
  'n': 1477,
  'DESCR': 'poker-8_vs_6',
  'n_minority': 17,
  'imbalance_ratio': 85.88235294117646,
  'data_loader': 'load_poker_8_vs_6'},
 {'name': 'poker-9_vs_7',
  'phenotype': 'poker',
  'citation_key': 'keel',
  'n_col': 10,
  'n_col_orig': 10,
  'n_col_non_unique_orig': 10,
  'n': 244,
  'DESCR': 'poker-9_vs_7',
  'n_minority': 8,
  'imbalance_ratio': 29.5,
  'data_loader': 'load_poker_9_vs_7'},
 {'name': 'saheart',
  'phenotype': 'saheart',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 462,
  'DESCR': 'saheart',
  'n_minority': 160,
  'imbalance_ratio': 1.8875,
  'data_loader': 'load_saheart'},
 {'name': 'SATIMAGE',
  'phenotype': 'SATIMAGE',
  'citation_key': 'krnn',
  'n_col': 36,
  'n_col_orig': 36,
  'n_col_non_unique_orig': 36,
  'n': 6435,
  'DESCR': 'SATIMAGE',
  'n_minority': 626,
  'imbalance_ratio': 9.279552715654953,
  'data_loader': 'load_satimage'},
 {'name': 'segment0',
  'phenotype': 'segment',
  'citation_key': 'keel',
  'n_col': 19,
  'n_col_orig': 19,
  'n_col_non_unique_orig': 18,
  'n': 2308,
  'DESCR': 'segment0',
  'n_minority': 329,
  'imbalance_ratio': 6.015197568389058,
  'data_loader': 'load_segment0'},
 {'name': 'shuttle-2_vs_5',
  'phenotype': 'shuttle',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 3316,
  'DESCR': 'shuttle-2_vs_5',
  'n_minority': 49,
  'imbalance_ratio': 66.6734693877551,
  'data_loader': 'load_shuttle_2_vs_5'},
 {'name': 'shuttle-6_vs_2-3',
  'phenotype': 'shuttle',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 230,
  'DESCR': 'shuttle-6_vs_2-3',
  'n_minority': 10,
  'imbalance_ratio': 22.0,
  'data_loader': 'load_shuttle_6_vs_2_3'},
 {'name': 'shuttle-c0-vs-c4',
  'phenotype': 'shuttle',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 1829,
  'DESCR': 'shuttle-c0-vs-c4',
  'n_minority': 123,
  'imbalance_ratio': 13.869918699186991,
  'data_loader': 'load_shuttle_c0_vs_c4'},
 {'name': 'shuttle-c2-vs-c4',
  'phenotype': 'shuttle',
  'citation_key': 'keel',
  'n_col': 9,
  'n_col_orig': 9,
  'n_col_non_unique_orig': 9,
  'n': 129,
  'DESCR': 'shuttle-c2-vs-c4',
  'n_minority': 6,
  'imbalance_ratio': 20.5,
  'data_loader': 'load_shuttle_c2_vs_c4'},
 {'name': 'SPECTF',
  'phenotype': 'SPECTF',
  'citation_key': 'krnn',
  'n_col': 44,
  'n_col_orig': 44,
  'n_col_non_unique_orig': 44,
  'n': 267,
  'DESCR': 'SPECTF',
  'n_minority': 212,
  'imbalance_ratio': 0.25943396226415094,
  'data_loader': 'load_spectf'}]
