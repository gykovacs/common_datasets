import sys

import numpy as np
import pandas as pd

from mldb._io import read_csv_data, read_arff_data, construct_return_set, references

__all__= ['load_ada',
            'load_cm1',
            'load_german',
            'load_hepatitis',
            'load_hiva',
            'load_hypothyroid',
            'load_kc1',
            'load_pc1',
            'load_satimage',
            'load_spectf',
            'load_sylva',
            'load_abalone_17_vs_7_8_9_10',
            'load_abalone_19_vs_10_11_12_13',
            'load_abalone_20_vs_8_9_10',
            'load_abalone_21_vs_8',
            'load_abalone_3_vs_11',
            'load_abalone19',
            'load_abalone9_18',
            'load_car_good',
            'load_car_vgood',
            'load_cleveland_0_vs_4',
            'load_dermatology_6',
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
            'load_flaref',
            'load_glass_0_1_4_6_vs_2',
            'load_glass_0_1_5_vs_2',
            'load_glass_0_1_6_vs_2',
            'load_glass_0_1_6_vs_5',
            'load_glass_0_4_vs_5',
            'load_glass_0_6_vs_5',
            'load_glass2',
            'load_glass4',
            'load_glass5',
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
            'load_led7digit_0_2_4_5_6_7_8_9_vs_1',
            'load_lymphography_normal_fibrosis',
            'load_page_blocks_1_3_vs_4',
            'load_poker_8_9_vs_5',
            'load_poker_8_9_vs_6',
            'load_poker_8_vs_6',
            'load_poker_9_vs_7',
            'load_shuttle_2_vs_5',
            'load_shuttle_6_vs_2_3',
            'load_shuttle_c0_vs_c4',
            'load_shuttle_c2_vs_c4',
            'load_vowel0',
            'load_winequality_red_3_vs_5',
            'load_winequality_red_4',
            'load_winequality_red_8_vs_6',
            'load_winequality_red_8_vs_6_7',
            'load_winequality_white_3_9_vs_5',
            'load_winequality_white_3_vs_7',
            'load_winequality_white_9_vs_4',
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
            'load_zoo_3',
            'load_ecoli_0_vs_1',
            'load_ecoli1',
            'load_ecoli2',
            'load_ecoli3',
            'load_glass_0_1_2_3_vs_4_5_6',
            'load_glass0',
            'load_glass1',
            'load_glass6',
            'load_habarman',
            'load_iris0',
            'load_new_thyroid1',
            'load_page_blocks0',
            'load_pima',
            'load_segment0',
            'load_vehicle0',
            'load_vehicle1',
            'load_vehicle2',
            'load_vehicle3',
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
            'load_spectfheart',
            'summary',
            'generate_summary',
            'get_data_loaders',
            'get_filtered_data_loaders',
            'get_references',
            'generate_summary_table']

def get_references():
    return references

def load_hiva(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db= read_csv_data('data/classification/hiva/hiva_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/classification/hiva/hiva_train.labels')
    db['target']= target
    
    return construct_return_set(db, "HIVA", return_X_y, encode, citation= 'krnn', name= "HIVA", verbose= verbose, onehot_threshold=onehot_threshold)

def load_hypothyroid(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db= read_csv_data('data/classification/hypothyroid/hypothyroid.data.txt')
    db.columns= ['target'] + list(db.columns[1:])
    
    return construct_return_set(db, "hypothyroid", return_X_y, encode, citation= 'krnn', name= "hypothyroid", verbose= verbose, onehot_threshold=onehot_threshold)

def load_sylva(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db= read_csv_data('data/classification/sylva/sylva_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/classification/sylva/sylva_train.labels')
    db['target']= target
    
    return construct_return_set(db, "sylva", return_X_y, encode, citation= 'krnn', name= "sylva", verbose= verbose, onehot_threshold=onehot_threshold)

def load_pc1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/pc1/pc1.arff')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "PC1", return_X_y, encode, citation= 'krnn', name= "PC1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_cm1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/cm1/cm1.arff.txt')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "CM1", return_X_y, encode, citation= 'krnn', name= "CM1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kc1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kc1/kc1.arff.txt')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "KC1", return_X_y, encode, citation= 'krnn', name= "KC1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_spectf(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db0= read_csv_data('data/classification/spect_f/SPECTF.train.txt')
    db1= read_csv_data('data/classification/spect_f/SPECTF.test.txt')
    db= pd.concat([db0, db1])
    db.columns= ['target'] + list(db.columns[1:])
    
    return construct_return_set(db, "SPECT_F", return_X_y, encode, citation= 'krnn', name= "SPECT_F", verbose= verbose, onehot_threshold=onehot_threshold)

def load_hepatitis(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db= read_csv_data('data/classification/hepatitis/hepatitis.data.txt')
    db.columns= ['target'] + list(db.columns[1:])

    return construct_return_set(db, "hepatitis", return_X_y, encode, citation= 'krnn', name= "hepatitis", verbose= verbose, onehot_threshold=onehot_threshold)

def load_vehicle(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db0= read_csv_data('data/classification/vehicle/xaa.dat.txt', sep= ' ', usecols= range(19))
    db1= read_csv_data('data/classification/vehicle/xab.dat.txt', sep= ' ', usecols= range(19))
    db2= read_csv_data('data/classification/vehicle/xac.dat.txt', sep= ' ', usecols= range(19))
    db3= read_csv_data('data/classification/vehicle/xad.dat.txt', sep= ' ', usecols= range(19))
    db4= read_csv_data('data/classification/vehicle/xae.dat.txt', sep= ' ', usecols= range(19))
    db5= read_csv_data('data/classification/vehicle/xaf.dat.txt', sep= ' ', usecols= range(19))
    db6= read_csv_data('data/classification/vehicle/xag.dat.txt', sep= ' ', usecols= range(19))
    db7= read_csv_data('data/classification/vehicle/xah.dat.txt', sep= ' ', usecols= range(19))
    db8= read_csv_data('data/classification/vehicle/xai.dat.txt', sep= ' ', usecols= range(19))
    
    db= pd.concat([db0, db1, db2, db3, db4, db5, db6, db7, db8])
    
    db.columns= list(db.columns[:-1]) + ['target']
    db.loc[db['target'] != 'van', 'target']= 'other'
    
    return construct_return_set(db, "vehicle", return_X_y, encode, citation= 'krnn', name= "vehicle", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ada(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db= read_csv_data('data/classification/ada/ada_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/classification/ada/ada_train.labels')
    db['target']= target
    
    return construct_return_set(db, "ADA", return_X_y, encode, citation= 'krnn', name= "ADA", verbose= verbose, onehot_threshold=onehot_threshold)

def load_german(return_X_y= False, encode= True, verbose=False, onehot_threshold=20):
    db= read_csv_data('data/classification/german/german.data.txt', sep= ' ')
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "german", return_X_y, encode, citation= 'krnn', name= "german", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db= read_csv_data('data/classification/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    db.loc[db['target'] != 3, 'target']= 0
    del db[db.columns[0]]
    
    return construct_return_set(db, "glass", return_X_y, encode, citation= 'krnn', name= "glass", verbose= verbose, onehot_threshold=onehot_threshold)

def load_satimage(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    db0= read_csv_data('data/classification/satimage/sat.trn.txt', sep= ' ')
    db1= read_csv_data('data/classification/satimage/sat.tst.txt', sep= ' ')
    db= pd.concat([db0, db1])
    db.columns= list(db.columns[:-1]) + ['target']
    db.loc[db['target'] != 4, 'target']= 0
    
    return construct_return_set(db, "SATIMAGE", return_X_y, encode, citation= 'krnn', name= "SATIMAGE", verbose= verbose, onehot_threshold=onehot_threshold)

def load_abalone_17_vs_7_8_9_10(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/abalone-17_vs_7-8-9-10/abalone-17_vs_7-8-9-10.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone_17_vs_7_8_9_10", return_X_y, encode, citation= 'keel', name= "abalone_17_vs_7_8_9_10", verbose= verbose, onehot_threshold=onehot_threshold)

def load_abalone_19_vs_10_11_12_13(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/abalone-19_vs_10-11-12-13/abalone-19_vs_10-11-12-13.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-19_vs_10-11-12-13", return_X_y, encode, citation= 'keel', name= "abalone-19_vs_10-11-12-13", verbose= verbose, onehot_threshold=onehot_threshold)

def load_abalone_20_vs_8_9_10(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/abalone-20_vs_8-9-10/abalone-20_vs_8-9-10.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-20_vs_8-9-10", return_X_y, encode, citation= 'keel', name= "abalone-20_vs_8-9-10", verbose= verbose, onehot_threshold=onehot_threshold)

def load_abalone_21_vs_8(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/abalone-21_vs_8/abalone-21_vs_8.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-21_vs_8", return_X_y, encode, citation= 'keel', name= "abalone-21_vs_8", verbose= verbose, onehot_threshold=onehot_threshold)

def load_abalone_3_vs_11(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/abalone-3_vs_11/abalone-3_vs_11.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-3_vs_11", return_X_y, encode, citation= 'keel', name= "abalone-3_vs_11", verbose= verbose, onehot_threshold=onehot_threshold)

def load_abalone19(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/abalone19/abalone19.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone19", return_X_y, encode, citation= 'keel', name= "abalone19", verbose= verbose, onehot_threshold=onehot_threshold)

def load_abalone9_18(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/abalone9-18/abalone9-18.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone9-18", return_X_y, encode, citation= 'keel', name= "abalone9-18", verbose= verbose, onehot_threshold=onehot_threshold)

def load_car_good(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/car-good/car-good.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "car_good", return_X_y, encode, citation= 'keel', name= "car_good", verbose= verbose, onehot_threshold=onehot_threshold)

def load_car_vgood(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/car-vgood/car-vgood.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "car-vgood", return_X_y, encode, citation= 'keel', name= "car-vgood", verbose= verbose, onehot_threshold=onehot_threshold)

def load_cleveland_0_vs_4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/cleveland-0_vs_4/cleveland-0_vs_4_no_null.dat')
    db= pd.DataFrame(data)
    db.loc[db['num'] == b'negative', 'num']= False
    db.loc[db['num'] == b'positive', 'num']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "cleveland-0_vs_4", return_X_y, encode, citation= 'keel', name= "cleveland-0_vs_4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_dermatology_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/dermatology-6/dermatology-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "dermatology-6", return_X_y, encode, citation= 'keel', name= "dermatology-6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_1_3_7_vs_2_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-1-3-7_vs_2-6/ecoli-0-1-3-7_vs_2-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-3-7_vs_2-6", return_X_y, encode, citation= 'keel', name= "ecoli-0-1-3-7_vs_2-6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_1_4_6_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-1-4-6_vs_5/ecoli-0-1-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-6_vs_5", return_X_y, encode, citation= 'keel', name= "ecoli-0-1-4-6_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_1_4_7_vs_2_3_5_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-1-4-7_vs_2-3-5-6/ecoli-0-1-4-7_vs_2-3-5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-7_vs_2-3-5-6", return_X_y, encode, citation= 'keel', name= "ecoli-0-1-4-7_vs_2-3-5-6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_1_4_7_vs_5_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-1-4-7_vs_5-6/ecoli-0-1-4-7_vs_5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-7_vs_5-6", return_X_y, encode, citation= 'keel', name= "ecoli-0-1-4-7_vs_5-6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_1_vs_2_3_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-1_vs_2-3-5/ecoli-0-1_vs_2-3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1_vs_2-3-5", return_X_y, encode, citation= 'keel', name= "ecoli-0-1_vs_2-3-5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_1_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-1_vs_5/ecoli-0-1_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1_vs_5", return_X_y, encode, citation= 'keel', name= "ecoli-0-1_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_2_3_4_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-2-3-4_vs_5/ecoli-0-2-3-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-2-3-4_vs_5", return_X_y, encode, citation= 'keel', name= "ecoli-0-2-3-4_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_2_6_7_vs_3_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-2-6-7_vs_3-5/ecoli-0-2-6-7_vs_3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-2-6-7_vs_3-5", return_X_y, encode, citation= 'keel', name= "ecoli-0-2-6-7_vs_3-5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_3_4_6_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-3-4-6_vs_5/ecoli-0-3-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4-6_vs_5", return_X_y, encode, citation= 'keel', name= "ecoli-0-3-4-6_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_3_4_7_vs_5_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-3-4-7_vs_5-6/ecoli-0-3-4-7_vs_5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4-7_vs_5-6", return_X_y, encode, citation= 'keel', name= "ecoli-0-3-4-7_vs_5-6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_3_4_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-3-4_vs_5/ecoli-0-3-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4_vs_5", return_X_y, encode, citation= 'keel', name= "ecoli-0-3-4_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_4_6_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-4-6_vs_5/ecoli-0-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-4-6_vs_5", return_X_y, encode, citation= 'keel', name= "ecoli-0-4-6_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_6_7_vs_3_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-6-7_vs_3-5/ecoli-0-6-7_vs_3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-6-7_vs_3-5", return_X_y, encode, citation= 'keel', name= "ecoli-0-6-7_vs_3-5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli_0_6_7_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0-6-7_vs_5/ecoli-0-6-7_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-6-7_vs_5", return_X_y, encode, citation= 'keel', name= "ecoli-0-6-7_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli4/ecoli4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli4", return_X_y, encode, citation= 'keel', name= "ecoli4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_flaref(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/flare-F/flare-F.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "flare-F", return_X_y, encode, citation= 'keel', name= "flare-F", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass_0_1_4_6_vs_2(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass-0-1-4-6_vs_2/glass-0-1-4-6_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-4-6_vs_2", return_X_y, encode, citation= 'keel', name= "glass-0-1-4-6_vs_2", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass_0_1_5_vs_2(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass-0-1-5_vs_2/glass-0-1-5_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-5_vs_2", return_X_y, encode, citation= 'keel', name= "glass-0-1-5_vs_2", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass_0_1_6_vs_2(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass-0-1-6_vs_2/glass-0-1-6_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-6_vs_2", return_X_y, encode, citation= 'keel', name= "glass-0-1-6_vs_2", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass_0_1_6_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass-0-1-6_vs_5/glass-0-1-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-6_vs_5", return_X_y, encode, citation= 'keel', name= "glass-0-1-6_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass_0_4_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass-0-4_vs_5/glass-0-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-4_vs_5", return_X_y, encode, citation= 'keel', name= "glass-0-4_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass_0_6_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass-0-6_vs_5/glass-0-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-6_vs_5", return_X_y, encode, citation= 'keel', name= "glass-0-6_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass2(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass2/glass2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass2", return_X_y, encode, citation= 'keel', name= "glass2", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass4/glass4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass4", return_X_y, encode, citation= 'keel', name= "glass4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass5/glass5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass5", return_X_y, encode, citation= 'keel', name= "glass5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kddcup_buffer_overflow_vs_back(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kddcup-buffer_overflow_vs_back/kddcup-buffer_overflow_vs_back.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-buffer_overflow_vs_back", return_X_y, encode, citation= 'keel', name= "kddcup-buffer_overflow_vs_back", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kddcup_guess_passwd_vs_satan(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kddcup-guess_passwd_vs_satan/kddcup-guess_passwd_vs_satan.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-guess_passwd_vs_satan", return_X_y, encode, citation= 'keel', name= "kddcup-guess_passwd_vs_satan", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kddcup_land_vs_portsweep(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kddcup-land_vs_portsweep/kddcup-land_vs_portsweep.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-land_vs_portsweep", return_X_y, encode, citation= 'keel', name= "kddcup-land_vs_portsweep", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kddcup_land_vs_satan(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kddcup-land_vs_satan/kddcup-land_vs_satan.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-land_vs_satan", return_X_y, encode, citation= 'keel', name= "kddcup-land_vs_satan", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kddcup_rootkit_imap_vs_back(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kddcup-rootkit-imap_vs_back/kddcup-rootkit-imap_vs_back.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-rootkit-imap_vs_back", return_X_y, encode, citation= 'keel', name= "kddcup-rootkit-imap_vs_back", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kr_vs_k_one_vs_fifteen(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kr-vs-k-one_vs_fifteen/kr-vs-k-one_vs_fifteen.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-one_vs_fifteen", return_X_y, encode, citation= 'keel', name= "kr-vs-k-one_vs_fifteen", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kr_vs_k_three_vs_eleven(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kr-vs-k-three_vs_eleven/kr-vs-k-three_vs_eleven.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-three_vs_eleven", return_X_y, encode, citation= 'keel', name= "kr-vs-k-three_vs_eleven", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kr_vs_k_zero_one_vs_draw(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kr-vs-k-zero-one_vs_draw/kr-vs-k-zero-one_vs_draw.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero-one_vs_draw", return_X_y, encode, citation= 'keel', name= "kr-vs-k-zero-one_vs_draw", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kr_vs_k_zero_vs_eight(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kr-vs-k-zero_vs_eight/kr-vs-k-zero_vs_eight.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero_vs_eight", return_X_y, encode, citation= 'keel', name= "kr-vs-k-zero_vs_eight", verbose= verbose, onehot_threshold=onehot_threshold)

def load_kr_vs_k_zero_vs_fifteen(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kr-vs-k-zero_vs_fifteen/kr-vs-k-zero_vs_fifteen.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero_vs_fifteen", return_X_y, encode, citation= 'keel', name= "kr-vs-k-zero_vs_fifteen", verbose= verbose, onehot_threshold=onehot_threshold)

def load_led7digit_0_2_4_5_6_7_8_9_vs_1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/led7digit-0-2-4-5-6-7-8-9_vs_1/led7digit-0-2-4-5-6-7-8-9_vs_1.dat')
    db= pd.DataFrame(data)
    db.loc[db['number'] == b'negative', 'number']= False
    db.loc[db['number'] == b'positive', 'number']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "led7digit-0-2-4-6-7-8-9_vs_1", return_X_y, encode, citation= 'keel', name= "led7digit-0-2-4-6-7-8-9_vs_1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_lymphography_normal_fibrosis(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/lymphography-normal-fibrosis/lymphography-normal-fibrosis.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "lymphography-normal-fibrosis", return_X_y, encode, citation= 'keel', name= "lymphography-normal-fibrosis", verbose= verbose, onehot_threshold=onehot_threshold)

def load_page_blocks_1_3_vs_4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/page-blocks-1-3_vs_4/page-blocks-1-3_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "page-blocks-1-3_vs_4", return_X_y, encode, citation= 'keel', name= "page-blocks-1-3_vs_4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_poker_8_9_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/poker-8-9_vs_5/poker-8-9_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8-9_vs_5", return_X_y, encode, citation= 'keel', name= "poker-8-9_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_poker_8_9_vs_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/poker-8-9_vs_6/poker-8-9_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8-9_vs_6", return_X_y, encode, citation= 'keel', name= "poker-8-9_vs_6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_poker_8_vs_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/poker-8_vs_6/poker-8_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8_vs_6", return_X_y, encode, citation= 'keel', name= "poker-8_vs_6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_poker_9_vs_7(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/poker-9_vs_7/poker-9_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-9_vs_7", return_X_y, encode, citation= 'keel', name= "poker-9_vs_7", verbose= verbose, onehot_threshold=onehot_threshold)

def load_shuttle_2_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/shuttle-2_vs_5/shuttle-2_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-2_vs_5", return_X_y, encode, citation= 'keel', name= "shuttle-2_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_shuttle_6_vs_2_3(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/shuttle-6_vs_2-3/shuttle-6_vs_2-3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-6_vs_2-3", return_X_y, encode, citation= 'keel', name= "shuttle-6_vs_2-3", verbose= verbose, onehot_threshold=onehot_threshold)

def load_shuttle_c0_vs_c4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/shuttle-c0-vs-c4/shuttle-c0-vs-c4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-c0-vs-c4", return_X_y, encode, citation= 'keel', name= "shuttle-c0-vs-c4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_shuttle_c2_vs_c4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/shuttle-c2-vs-c4/shuttle-c2-vs-c4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-c2-vs-c4", return_X_y, encode, citation= 'keel', name= "shuttle-c2-vs-c4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_vowel0(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/vowel0/vowel0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vowel0", return_X_y, encode, citation= 'keel', name= "vowel0", verbose= verbose, onehot_threshold=onehot_threshold)

def load_winequality_red_3_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/winequality-red-3_vs_5/winequality-red-3_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-3_vs_5", return_X_y, encode, citation= 'keel', name= "winequality-red-3_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_winequality_red_4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/winequality-red-4/winequality-red-4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-4", return_X_y, encode, citation= 'keel', name= "winequality-red-4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_winequality_red_8_vs_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/winequality-red-8_vs_6/winequality-red-8_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-8_vs_6", return_X_y, encode, citation= 'keel', name= "winequality-red-8_vs_6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_winequality_red_8_vs_6_7(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/winequality-red-8_vs_6-7/winequality-red-8_vs_6-7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-8_vs_6-7", return_X_y, encode, citation= 'keel', name= "winequality-red-8_vs_6-7", verbose= verbose, onehot_threshold=onehot_threshold)

def load_winequality_white_3_9_vs_5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/winequality-white-3-9_vs_5/winequality-white-3-9_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-3-9_vs_5", return_X_y, encode, citation= 'keel', name= "winequality-white-3-9_vs_5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_winequality_white_3_vs_7(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/winequality-white-3_vs_7/winequality-white-3_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-3_vs_7", return_X_y, encode, citation= 'keel', name= "winequality-white-3_vs_7", verbose= verbose, onehot_threshold=onehot_threshold)

def load_winequality_white_9_vs_4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/winequality-white-9_vs_4/winequality-white-9_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-9_vs_4", return_X_y, encode, citation= 'keel', name= "winequality-white-9_vs_4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_0_2_5_6_vs_3_7_8_9(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-0-2-5-6_vs_3-7-8-9/yeast-0-2-5-6_vs_3-7-8-9.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-2-5-6_vs_3-7-8-9", return_X_y, encode, citation= 'keel', name= "yeast-0-2-5-6_vs_3-7-8-9", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_0_2_5_7_9_vs_3_6_8(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-0-2-5-7-9_vs_3-6-8/yeast-0-2-5-7-9_vs_3-6-8.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-2-5-7-9_vs_3-6-8", return_X_y, encode, citation= 'keel', name= "yeast-0-2-5-7-9_vs_3-6-8", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_0_3_5_9_vs_7_8(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-0-3-5-9_vs_7-8/yeast-0-3-5-9_vs_7-8.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-3-5-9_vs_7-8", return_X_y, encode, citation= 'keel', name= "yeast-0-3-5-9_vs_7-8", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_0_5_6_7_9_vs_4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-0-5-6-7-9_vs_4/yeast-0-5-6-7-9_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-5-6-7-9_vs_4", return_X_y, encode, citation= 'keel', name= "yeast-0-5-6-7-9_vs_4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_1_2_8_9_vs_7(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-1-2-8-9_vs_7/yeast-1-2-8-9_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1-2-8-9_vs_7", return_X_y, encode, citation= 'keel', name= "yeast-1-2-8-9_vs_7", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_1_4_5_8_vs_7(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-1-4-5-8_vs_7/yeast-1-4-5-8_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1-4-5-8_vs_7", return_X_y, encode, citation= 'keel', name= "yeast-1-4-5-8_vs_7", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_1_vs_7(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-1_vs_7/yeast-1_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1_vs_7", return_X_y, encode, citation= 'keel', name= "yeast-1_vs_7", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_2_vs_4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-2_vs_4/yeast-2_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-2_vs_4", return_X_y, encode, citation= 'keel', name= "yeast-2_vs_4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast_2_vs_8(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast-2_vs_8/yeast-2_vs_8.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-2_vs_8", return_X_y, encode, citation= 'keel', name= "yeast-2_vs_8", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast4(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast4/yeast4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast4", return_X_y, encode, citation= 'keel', name= "yeast4", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast5(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast5/yeast5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast5", return_X_y, encode, citation= 'keel', name= "yeast5", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast6/yeast6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast6", return_X_y, encode, citation= 'keel', name= "yeast6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_zoo_3(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/zoo-3/zoo-3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "zoo-3", return_X_y, encode, citation= 'keel', name= "zoo-3", verbose= verbose, onehot_threshold=onehot_threshold)

#########################

def load_ecoli_0_vs_1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli-0_vs_1/ecoli-0_vs_1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'class']= False
    db.loc[db['Class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0_vs_1", return_X_y, encode, citation= 'keel', name= "ecoli-0_vs_1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli1/ecoli1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli1", return_X_y, encode, citation= 'keel', name= "ecoli1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli2(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli2/ecoli2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli2", return_X_y, encode, citation= 'keel', name= "ecoli2", verbose= verbose, onehot_threshold=onehot_threshold)

def load_ecoli3(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ecoli3/ecoli3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli3", return_X_y, encode, citation= 'keel', name= "ecoli3", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass_0_1_2_3_vs_4_5_6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass-0-1-2-3_vs_4-5-6/glass-0-1-2-3_vs_4-5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-2-3_vs_4-5-6", return_X_y, encode, citation= 'keel', name= "glass-0-1-2-3_vs_4-5-6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass0(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass0/glass0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass0", return_X_y, encode, citation= 'keel', name= "glass0", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass1/glass1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass1", return_X_y, encode, citation= 'keel', name= "glass1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_glass6(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/glass6/glass6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass6", return_X_y, encode, citation= 'keel', name= "glass6", verbose= verbose, onehot_threshold=onehot_threshold)

def load_habarman(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/haberman/haberman.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "habarman", return_X_y, encode, citation= 'keel', name= "habarman", verbose= verbose, onehot_threshold=onehot_threshold)

def load_iris0(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/iris0/iris0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "iris0", return_X_y, encode, citation= 'keel', name= "iris0", verbose= verbose, onehot_threshold=onehot_threshold)

def load_new_thyroid1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/new_thyroid1/new-thyroid1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "new_thyroid1", return_X_y, encode, citation= 'keel', name= "new_thyroid1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_new_thyroid2(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/new_thyroid2/new_thyroid2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "new_thyroid2", return_X_y, encode, citation= 'keel', name= "new_thyroid2", verbose= verbose, onehot_threshold=onehot_threshold)

def load_page_blocks0(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/page-blocks0/page-blocks0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "page_blocks0", return_X_y, encode, citation= 'keel', name= "page_blocks0", verbose= verbose, onehot_threshold=onehot_threshold)

def load_pima(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/pima/pima.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "pima", return_X_y, encode, citation= 'keel', name= "pima", verbose= verbose, onehot_threshold=onehot_threshold)

def load_segment0(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/segment0/segment0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "segment0", return_X_y, encode, citation= 'keel', name= "segment0", verbose= verbose, onehot_threshold=onehot_threshold)

def load_vehicle0(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/vehicle0/vehicle0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle0", return_X_y, encode, citation= 'keel', name= "vehicle0", verbose= verbose, onehot_threshold=onehot_threshold)

def load_vehicle1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/vehicle1/vehicle1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle1", return_X_y, encode, citation= 'keel', name= "vehicle1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_vehicle2(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/vehicle2/vehicle2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle2", return_X_y, encode, citation= 'keel', name= "vehicle2", verbose= verbose, onehot_threshold=onehot_threshold)

def load_vehicle3(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/vehicle3/vehicle3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle3", return_X_y, encode, citation= 'keel', name= "vehicle3", verbose= verbose, onehot_threshold=onehot_threshold)

def load_wisconsin(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/wisconsin/wisconsin.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wisconsin", return_X_y, encode, citation= 'keel', name= "wisconsin", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast1(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast1/yeast1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast1", return_X_y, encode, citation= 'keel', name= "yeast1", verbose= verbose, onehot_threshold=onehot_threshold)

def load_yeast3(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/yeast3/yeast3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast3", return_X_y, encode, citation= 'keel', name= "yeast3", verbose= verbose, onehot_threshold=onehot_threshold)

def load_mammographic(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/mammographic/mammographic.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "mammographic", return_X_y, encode, citation= 'keel', name='mammographic', verbose=verbose, onehot_threshold=onehot_threshold)

def load_bupa(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/bupa/bupa.dat')
    db= pd.DataFrame(data)
    db.loc[db[db.columns[-1]] == 1, db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == 2, db.columns[-1]]= 1
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "bupa", return_X_y, encode, citation= 'keel', name="bupa", verbose=verbose, onehot_threshold=onehot_threshold)

def load_monk_2(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/monk-2/monk-2.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "monk-2", return_X_y, encode, citation= 'keel', name="monk-2", verbose=verbose, onehot_threshold=onehot_threshold)

def load_appendicitis(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/appendicitis/appendicitis.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "appendicitis", return_X_y, encode, citation= 'keel', name="appendicitis", verbose=verbose, onehot_threshold=onehot_threshold)

def load_saheart(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/saheart/saheart.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "saheart", return_X_y, encode, citation= 'keel', name="saheart", verbose=verbose, onehot_threshold=onehot_threshold)

def load_australian(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/australian/australian.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "australian", return_X_y, encode, citation= 'keel', name="australian", verbose=verbose, onehot_threshold=onehot_threshold)

def load_crx(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/crx/crx.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "crx", return_X_y, encode, citation= 'keel', name="crx", verbose=verbose, onehot_threshold=onehot_threshold)

def load_lymphography(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/lymphography/lymphography.dat')
    db= pd.DataFrame(data)
    db.loc[db[db.columns[-1]] == b'metastases', db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == b'malign_lymph', db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == b'normal', db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == b'fibrosis', db.columns[-1]]= 1
    db[db.columns[-1]]= db[db.columns[-1]].astype(int)
    db.columns= list(db.columns[:-1]) + ['target']
    
    print(np.unique(db.values[:,-1], return_counts=True))

    return construct_return_set(db, "lymphography", return_X_y, encode, citation= 'keel', name="lymphography", verbose=verbose, onehot_threshold=onehot_threshold)

def load_wdbc(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/wdbc/wdbc.dat')
    db= pd.DataFrame(data)
    db.loc[db[db.columns[-1]] == 'M', db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == 'B', db.columns[-1]]= 1
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wdbc", return_X_y, encode, citation= 'keel', name="wdbc", verbose=verbose, onehot_threshold=onehot_threshold)

def load_ionosphere(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/ionosphere/ionosphere.dat')
    db= pd.DataFrame(data)
    db.loc[db[db.columns[-1]] == 'g', db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == 'b', db.columns[-1]]= 1
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ionosphere", return_X_y, encode, citation= 'keel', name="ionosphere", verbose=verbose, onehot_threshold=onehot_threshold)

def load_spectfheart(return_X_y= False, encode= True, verbose=False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/spectfheart/spectfheart.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "spectfheart", return_X_y, encode, citation= 'keel', name="spectfheart", verbose=verbose, onehot_threshold=onehot_threshold)

def generate_summary():
    results= []
    
    for func_name in __all__:
        if func_name.startswith('load_'):
            data_encoded= globals()[func_name](return_X_y= False, encode= True)
            
            X= data_encoded['data']
            
            X_min= X[data_encoded['target'] == 1]
            X_maj= X[data_encoded['target'] == 0]
            
            result= {'loader_function': globals()[func_name],
                        'name': data_encoded['name'],
                        'n': len(data_encoded['data']),
                        'n_attr_raw': len(data_encoded['data_raw'][0]),
                        'n_attr_encoded': len(data_encoded['data'][0]),
                        'n_minority': np.sum(data_encoded['target'] == 1),
                        'imbalance_ratio': np.sum(data_encoded['target'] == 0)/np.sum(data_encoded['target'] == 1)}
            
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler
            nn= NearestNeighbors(n_neighbors= 2)
            
            dist, ind= nn.fit(X_min).kneighbors(X_min)
            mean_min_dist= np.mean(dist[:,1])
            dist, ind= nn.fit(X_maj).kneighbors(X_maj)
            mean_maj_dist= np.mean(dist[:,1])
            
            result['mean_min_dist']= mean_min_dist
            result['mean_maj_dist']= mean_maj_dist
            result['imbalance_ratio_dist']= mean_maj_dist/mean_min_dist
            
            result['reference_key']= data_encoded['citation']
            
            results.append(result)
    
    df_results= pd.DataFrame(results)

    return df_results

def summary():
    summary_df= pd.DataFrame(summary_table.copy(), columns=summary_columns)
    summary_df['loader_function']= summary_df['loader_function'].apply(lambda x: globals()[x])
    return summary_df

def generate_summary_table():
    results= generate_summary()
    results['loader_function']= results['loader_function'].apply(lambda x: x.__name__)

    return results.columns, results.values

def get_filtered_data_loaders(n_attr_encoded_bounds= [1, 5000],
                              n_attr_raw_bounds= [1, 5000],
                              n_bounds= [1, 10000],
                              n_minority_bounds= [1, 10000],
                              imbalance_ratio_bounds= [0, 1000]):
    descriptors= summary()
    return descriptors[(descriptors['n'] >= n_bounds[0]) & (descriptors['n'] < n_bounds[1]) & 
                       (descriptors['n_attr_encoded'] >= n_attr_encoded_bounds[0]) & (descriptors['n_attr_encoded'] < n_attr_encoded_bounds[1]) & 
                       (descriptors['n_attr_raw'] >= n_attr_raw_bounds[0]) & (descriptors['n_attr_raw'] < n_attr_raw_bounds[1]) &
                       (descriptors['imbalance_ratio'] >= imbalance_ratio_bounds[0]) & (descriptors['imbalance_ratio'] < imbalance_ratio_bounds[1]) &
                       (descriptors['n_minority'] >= n_minority_bounds[0]) & (descriptors['n_minority'] < n_minority_bounds[1])]['loader_function'].values

def get_data_loaders(subset='all'):
    """
    Args:
        subset (str): 'all'/'study'/'small'/'tiny'
    """
    
    n_attr_encoded_bounds= [1, 5000]
    n_attr_raw_bounds= [1, 5000]
    n_bounds= [1, 10000]
    n_minority_bounds= [1, 10000]
    imbalance_ratio_bounds= [0, 1000]
    
    if subset == 'study':
        n_attr_encoded_bounds[1]= 100
        n_bounds[1]= 4000
    elif subset == 'small':
        n_attr_encoded_bounds[1]= 100
        n_bounds[1]= 1000
    elif subset == 'tiny':
        n_bounds[1]= 120
    
    return get_filtered_data_loaders(n_attr_encoded_bounds= n_attr_encoded_bounds,
                                    n_attr_raw_bounds= n_attr_raw_bounds,
                                    n_bounds= n_bounds,
                                    n_minority_bounds= n_minority_bounds,
                                    imbalance_ratio_bounds= imbalance_ratio_bounds)

summary_columns= ['n_attr_encoded', 'imbalance_ratio', 'imbalance_ratio_dist', 'n',
        'loader_function', 'mean_maj_dist', 'mean_min_dist', 'n_minority',
        'name', 'n_attr_raw', 'reference_key']

summary_table= np.array([
       [3.03, 0.94, 'load_ada', 1.89, 2.02, 4147, 47, 48, 1029, 'ADA', 'krnn'],
       [9.16, 0.33, 'load_cm1', 0.75, 2.25, 498, 21, 21, 49, 'CM1', 'krnn'],
       [2.33, 0.87, 'load_german', 5.51, 6.34, 1000, 59, 20, 300, 'german', 'krnn'],
       [3.84, 0.71, 'load_hepatitis', 4.23, 5.96, 155, 44, 19, 32, 'hepatitis', 'krnn'],
       [27.48, 0.7, 'load_hiva', 26.32, 37.52, 3845, 1617, 1617, 135, 'HIVA', 'krnn'],
       [19.95, 0.59, 'load_hypothyroid', 0.93, 1.58, 3163, 34, 25, 151, 'hypothyroid', 'krnn'],
       [5.47, 0.24, 'load_kc1', 0.35, 1.45, 2109, 21, 21, 326, 'KC1', 'krnn'],
       [13.4, 0.26, 'load_pc1', 0.61, 2.38, 1109, 21, 21, 77, 'PC1', 'krnn'],
       [9.28, 1.04, 'load_satimage', 1.24, 1.19, 6435, 36, 36, 626, 'SATIMAGE', 'krnn'],
       [3.85, 1.39, 'load_spectf', 4.9, 3.52, 267, 44, 44, 55, 'SPECT_F', 'krnn'],
       [15.26, 0.83, 'load_sylva', 9.38, 11.35, 13086, 212, 216, 805, 'sylva', 'krnn'],
       [39.31, 0.36, 'load_abalone_17_vs_7_8_9_10', 0.34, 0.92, 2338, 10, 8, 58, 'abalone_17_vs_7_8_9_10', 'keel'],
       [49.69, 0.44, 'load_abalone_19_vs_10_11_12_13', 0.43, 0.99, 1622, 10, 8, 32, 'abalone-19_vs_10-11-12-13', 'keel'],
       [72.69, 0.32, 'load_abalone_20_vs_8_9_10', 0.38, 1.2, 1916, 10, 8, 26, 'abalone-20_vs_8-9-10', 'keel'],
       [40.5, 0.23, 'load_abalone_21_vs_8', 0.47, 2.04, 581, 10, 8, 14, 'abalone-21_vs_8', 'keel'],
       [32.47, 1.76, 'load_abalone_3_vs_11', 0.46, 0.26, 502, 10, 8, 15, 'abalone-3_vs_11', 'keel'],
       [129.44, 0.35, 'load_abalone19', 0.28, 0.81, 4174, 10, 8, 32, 'abalone19', 'keel'],
       [16.4, 0.35, 'load_abalone9_18', 0.48, 1.38, 731, 10, 8, 42, 'abalone9-18', 'keel'],
       [24.04, 1.0, 'load_car_good', 3.0, 3.0, 1728, 21, 6, 69, 'car_good', 'keel'],
       [25.58, 1.0, 'load_car_vgood', 3.0, 3.0, 1728, 21, 6, 65, 'car-vgood', 'keel'],
       [12.62, 0.58, 'load_cleveland_0_vs_4', 2.12, 3.65, 177, 13, 13, 13, 'cleveland-0_vs_4', 'keel'],
       [16.9, 0.83, 'load_dermatology_6', 3.11, 3.76, 358, 34, 34, 20, 'dermatology-6', 'keel'],
       [39.14, 0.54, 'load_ecoli_0_1_3_7_vs_2_6', 0.71, 1.32, 281, 7, 7, 7, 'ecoli-0-1-3-7_vs_2-6', 'keel'],
       [13.0, 0.46, 'load_ecoli_0_1_4_6_vs_5', 0.62, 1.36, 280, 6, 6, 20, 'ecoli-0-1-4-6_vs_5', 'keel'],
       [10.59, 0.34, 'load_ecoli_0_1_4_7_vs_2_3_5_6', 0.6, 1.78, 336, 7, 7, 29, 'ecoli-0-1-4-7_vs_2-3-5-6', 'keel'],
       [12.28, 0.56, 'load_ecoli_0_1_4_7_vs_5_6', 0.61, 1.08, 332, 6, 6, 25, 'ecoli-0-1-4-7_vs_5-6', 'keel'],
       [9.17, 0.34, 'load_ecoli_0_1_vs_2_3_5', 0.66, 1.97, 244, 7, 7, 24, 'ecoli-0-1_vs_2-3-5', 'keel'],
       [11.0, 0.42, 'load_ecoli_0_1_vs_5', 0.68, 1.61, 240, 6, 6, 20, 'ecoli-0-1_vs_5', 'keel'],
       [9.1, 0.48, 'load_ecoli_0_2_3_4_vs_5', 0.7, 1.45, 202, 7, 7, 20, 'ecoli-0-2-3-4_vs_5', 'keel'],
       [9.18, 0.32, 'load_ecoli_0_2_6_7_vs_3_5', 0.65, 2.05, 224, 7, 7, 22, 'ecoli-0-2-6-7_vs_3-5', 'keel'],
       [9.25, 0.52, 'load_ecoli_0_3_4_6_vs_5', 0.7, 1.35, 205, 7, 7, 20, 'ecoli-0-3-4-6_vs_5', 'keel'],
       [9.28, 0.61, 'load_ecoli_0_3_4_7_vs_5_6', 0.68, 1.12, 257, 7, 7, 25, 'ecoli-0-3-4-7_vs_5-6', 'keel'],
       [9.0, 0.48, 'load_ecoli_0_3_4_vs_5', 0.7, 1.45, 200, 7, 7, 20, 'ecoli-0-3-4_vs_5', 'keel'],
       [9.15, 0.45, 'load_ecoli_0_4_6_vs_5', 0.62, 1.38, 203, 6, 6, 20, 'ecoli-0-4-6_vs_5', 'keel'],
       [9.09, 0.32, 'load_ecoli_0_6_7_vs_3_5', 0.65, 2.05, 222, 7, 7, 22, 'ecoli-0-6-7_vs_3-5', 'keel'],
       [10.0, 0.45, 'load_ecoli_0_6_7_vs_5', 0.66, 1.46, 220, 6, 6, 20, 'ecoli-0-6-7_vs_5', 'keel'],
       [15.8, 0.57, 'load_ecoli4', 0.63, 1.1, 336, 7, 7, 20, 'ecoli4', 'keel'],
       [23.79, 0.17, 'load_flaref', 0.76, 4.53, 1066, 37, 11, 43, 'flare-F', 'keel'],
       [11.06, 0.87, 'load_glass_0_1_4_6_vs_2', 0.84, 0.96, 205, 9, 9, 17, 'glass-0-1-4-6_vs_2', 'keel'],
       [9.12, 0.82, 'load_glass_0_1_5_vs_2', 0.98, 1.19, 172, 9, 9, 17, 'glass-0-1-5_vs_2', 'keel'],
       [10.29, 0.87, 'load_glass_0_1_6_vs_2', 0.89, 1.03, 192, 9, 9, 17, 'glass-0-1-6_vs_2', 'keel'],
       [19.44, 0.46, 'load_glass_0_1_6_vs_5', 0.85, 1.84, 184, 9, 9, 9, 'glass-0-1-6_vs_5', 'keel'],
       [9.22, 0.42, 'load_glass_0_4_vs_5', 0.83, 1.97, 92, 9, 9, 9, 'glass-0-4_vs_5', 'keel'],
       [11.0, 0.44, 'load_glass_0_6_vs_5', 0.86, 1.94, 108, 9, 9, 9, 'glass-0-6_vs_5', 'keel'],
       [11.59, 0.97, 'load_glass2', 0.87, 0.89, 214, 9, 9, 17, 'glass2', 'keel'],
       [15.46, 0.45, 'load_glass4', 0.81, 1.78, 214, 9, 9, 13, 'glass4', 'keel'],
       [22.78, 0.45, 'load_glass5', 0.81, 1.81, 214, 9, 9, 9, 'glass5', 'keel'],
       [73.43, 0.02, 'load_kddcup_buffer_overflow_vs_back', 0.18, 12.11, 2233, 34, 41, 30, 'kddcup-buffer_overflow_vs_back', 'keel'],
       [29.98, 0.1, 'load_kddcup_guess_passwd_vs_satan', 0.29, 2.85, 1642, 39, 41, 53, 'kddcup-guess_passwd_vs_satan', 'keel'],
       [49.52, 0.19, 'load_kddcup_land_vs_portsweep', 0.34, 1.8, 1061, 32, 41, 21, 'kddcup-land_vs_portsweep', 'keel'],
       [75.67, 0.17, 'load_kddcup_land_vs_satan', 0.31, 1.79, 1610, 33, 41, 21, 'kddcup-land_vs_satan', 'keel'],
       [100.14, 0.01, 'load_kddcup_rootkit_imap_vs_back', 0.16, 28.31, 2225, 43, 41, 22, 'kddcup-rootkit-imap_vs_back', 'keel'],
       [27.77, 0.83, 'load_kr_vs_k_one_vs_fifteen', 3.31, 3.99, 2244, 39, 6, 78, 'kr-vs-k-one_vs_fifteen', 'keel'],
       [35.23, 1.0, 'load_kr_vs_k_three_vs_eleven', 3.48, 3.47, 2935, 40, 6, 81, 'kr-vs-k-three_vs_eleven', 'keel'],
       [26.63, 0.84, 'load_kr_vs_k_zero_one_vs_draw', 3.13, 3.73, 2901, 40, 6, 105, 'kr-vs-k-zero-one_vs_draw', 'keel'],
       [53.07, 0.94, 'load_kr_vs_k_zero_vs_eight', 3.31, 3.52, 1460, 40, 6, 27, 'kr-vs-k-zero_vs_eight', 'keel'],
       [80.22, 0.79, 'load_kr_vs_k_zero_vs_fifteen', 3.3, 4.2, 2193, 39, 6, 27, 'kr-vs-k-zero_vs_fifteen', 'keel'],
       [10.97, 0.25, 'load_led7digit_0_2_4_5_6_7_8_9_vs_1', 0.14, 0.54, 443, 7, 7, 37, 'led7digit-0-2-4-6-7-8-9_vs_1', 'keel'],
       [23.67, 0.48, 'load_lymphography_normal_fibrosis', 4.35, 8.97, 148, 38, 18, 6, 'lymphography-normal-fibrosis', 'keel'],
       [15.86, 0.27, 'load_page_blocks_1_3_vs_4', 0.3, 1.11, 472, 10, 10, 28, 'page-blocks-1-3_vs_4', 'keel'],
       [82.0, 0.38, 'load_poker_8_9_vs_5', 0.68, 1.8, 2075, 10, 10, 25, 'poker-8-9_vs_5', 'keel'],
       [58.4, 0.75, 'load_poker_8_9_vs_6', 1.36, 1.82, 1485, 10, 10, 25, 'poker-8-9_vs_6', 'keel'],
       [85.88, 0.77, 'load_poker_8_vs_6', 1.36, 1.77, 1477, 10, 10, 17, 'poker-8_vs_6', 'keel'],
       [29.5, 0.61, 'load_poker_9_vs_7', 1.58, 2.57, 244, 10, 10, 8, 'poker-9_vs_7', 'keel'],
       [66.67, 0.42, 'load_shuttle_2_vs_5', 0.08, 0.18, 3316, 9, 9, 49, 'shuttle-2_vs_5', 'keel'],
       [22.0, 0.12, 'load_shuttle_6_vs_2_3', 0.25, 2.04, 230, 9, 9, 10, 'shuttle-6_vs_2-3', 'keel'],
       [13.87, 0.15, 'load_shuttle_c0_vs_c4', 0.14, 0.9, 1829, 9, 9, 123, 'shuttle-c0-vs-c4', 'keel'],
       [20.5, 0.36, 'load_shuttle_c2_vs_c4', 0.41, 1.12, 129, 9, 9, 6, 'shuttle-c2-vs-c4', 'keel'],
       [9.98, 1.14, 'load_vowel0', 0.57, 0.5, 988, 13, 13, 90, 'vowel0', 'keel'],
       [68.1, 0.27, 'load_winequality_red_3_vs_5', 0.83, 3.05, 691, 11, 11, 10, 'winequality-red-3_vs_5', 'keel'],
       [29.17, 0.35, 'load_winequality_red_4', 0.74, 2.16, 1599, 11, 11, 53, 'winequality-red-4', 'keel'],
       [35.44, 0.43, 'load_winequality_red_8_vs_6', 0.89, 2.09, 656, 11, 11, 18, 'winequality-red-8_vs_6', 'keel'],
       [46.5, 0.41, 'load_winequality_red_8_vs_6_7', 0.84, 2.06, 855, 11, 11, 18, 'winequality-red-8_vs_6-7', 'keel'],
       [58.28, 0.21, 'load_winequality_white_3_9_vs_5', 0.76, 3.72, 1482, 11, 11, 25, 'winequality-white-3-9_vs_5', 'keel'],
       [44.0, 0.17, 'load_winequality_white_3_vs_7', 0.83, 4.84, 900, 11, 11, 20, 'winequality-white-3_vs_7', 'keel'],
       [32.6, 0.64, 'load_winequality_white_9_vs_4', 1.41, 2.21, 168, 11, 11, 5, 'winequality-white-9_vs_4', 'keel'],
       [9.14, 0.63, 'load_yeast_0_2_5_6_vs_3_7_8_9', 0.69, 1.09, 1004, 8, 8, 99, 'yeast-0-2-5-6_vs_3-7-8-9', 'keel'],
       [9.14, 0.74, 'load_yeast_0_2_5_7_9_vs_3_6_8', 0.69, 0.93, 1004, 8, 8, 99, 'yeast-0-2-5-7-9_vs_3-6-8', 'keel'],
       [9.12, 0.67, 'load_yeast_0_3_5_9_vs_7_8', 0.86, 1.28, 506, 8, 8, 50, 'yeast-0-3-5-9_vs_7-8', 'keel'],
       [9.35, 0.63, 'load_yeast_0_5_6_7_9_vs_4', 0.83, 1.33, 528, 8, 8, 51, 'yeast-0-5-6-7-9_vs_4', 'keel'],
       [30.57, 0.56, 'load_yeast_1_2_8_9_vs_7', 0.77, 1.38, 947, 8, 8, 30, 'yeast-1-2-8-9_vs_7', 'keel'],
       [22.1, 0.66, 'load_yeast_1_4_5_8_vs_7', 0.86, 1.3, 693, 8, 8, 30, 'yeast-1-4-5-8_vs_7', 'keel'],
       [14.3, 0.7, 'load_yeast_1_vs_7', 0.96, 1.37, 459, 7, 7, 30, 'yeast-1_vs_7', 'keel'],
       [9.08, 0.53, 'load_yeast_2_vs_4', 0.74, 1.41, 514, 8, 8, 51, 'yeast-2_vs_4', 'keel'],
       [23.1, 0.44, 'load_yeast_2_vs_8', 0.76, 1.74, 482, 8, 8, 20, 'yeast-2_vs_8', 'keel'],
       [28.1, 0.52, 'load_yeast4', 0.68, 1.31, 1484, 8, 8, 51, 'yeast4', 'keel'],
       [32.73, 0.75, 'load_yeast5', 0.68, 0.9, 1484, 8, 8, 44, 'yeast5', 'keel'],
       [41.4, 0.73, 'load_yeast6', 0.68, 0.94, 1484, 8, 8, 35, 'yeast6', 'keel'],
       [19.2, 0.31, 'load_zoo_3', 1.09, 3.47, 101, 21, 16, 5, 'zoo-3', 'keel'],
       [1.86, 0.56, 'load_ecoli_0_vs_1', 0.61, 1.08, 220, 7, 8, 77, 'ecoli-0_vs_1', 'keel'],
       [3.36, 0.78, 'load_ecoli1', 0.65, 0.83, 336, 7, 7, 77, 'ecoli1', 'keel'],
       [5.46, 1.02, 'load_ecoli2', 0.66, 0.65, 336, 7, 7, 52, 'ecoli2', 'keel'],
       [8.6, 0.76, 'load_ecoli3', 0.66, 0.87, 336, 7, 7, 35, 'ecoli3', 'keel'],
       [3.2, 0.46, 'load_glass_0_1_2_3_vs_4_5_6', 0.68, 1.47, 214, 9, 9, 51, 'glass-0-1-2-3_vs_4-5-6', 'keel'],
       [2.06, 1.99, 'load_glass0', 1.04, 0.52, 214, 9, 9, 70, 'glass0', 'keel'],
       [1.82, 0.92, 'load_glass1', 0.86, 0.94, 214, 9, 9, 76, 'glass1', 'keel'],
       [6.38, 0.51, 'load_glass6', 0.78, 1.53, 214, 9, 9, 29, 'glass6', 'keel'],
       [2.78, 0.52, 'load_habarman', 0.28, 0.53, 306, 3, 3, 81, 'habarman', 'keel'],
       [2.0, 1.33, 'load_iris0', 0.33, 0.25, 150, 4, 4, 50, 'iris0', 'keel'],
       [5.14, 0.55, 'load_new_thyroid1', 0.4, 0.72, 215, 5, 5, 35, 'new_thyroid1', 'keel'],
       [8.79, 0.2, 'load_page_blocks0', 0.16, 0.77, 5472, 10, 10, 559, 'page_blocks0', 'keel'],
       [1.87, 0.75, 'load_pima', 1.02, 1.36, 768, 8, 8, 268, 'pima', 'keel'],
       [6.02, 1.82, 'load_segment0', 0.47, 0.26, 2308, 18, 19, 329, 'segment0', 'keel'],
       [3.25, 0.99, 'load_vehicle0', 1.18, 1.2, 846, 18, 18, 199, 'vehicle0', 'keel'],
       [2.9, 0.88, 'load_vehicle1', 1.2, 1.37, 846, 18, 18, 217, 'vehicle1', 'keel'],
       [2.88, 1.09, 'load_vehicle2', 1.2, 1.1, 846, 18, 18, 218, 'vehicle2', 'keel'],
       [2.99, 0.88, 'load_vehicle3', 1.2, 1.37, 846, 18, 18, 212, 'vehicle3', 'keel'],
       [1.86, 0.16, 'load_wisconsin', 0.25, 1.54, 683, 9, 9, 239, 'wisconsin', 'keel'],
       [2.46, 0.8, 'load_yeast1', 0.69, 0.87, 1484, 8, 8, 429, 'yeast1', 'keel'],
       [8.1, 0.79, 'load_yeast3', 0.68, 0.86, 1484, 8, 8, 163, 'yeast3', 'keel'],
       [1.06, 0.71, 'load_mammographic', 0.17, 0.24, 830, 5, 5, 403, 'mammographic', 'keel'],
       [1.38, 1.07, 'load_bupa', 0.98, 0.92, 345, 6, 6, 145, 'bupa', 'keel'],
       [1.12, 0.88, 'load_monk_2', 0.91, 1.03, 432, 6, 6, 204, 'monk-2', 'keel'],
       [4.05, 0.7, 'load_appendicitis', 0.82, 1.16, 106, 7, 7, 21, 'appendicitis', 'keel'],
       [1.89, 0.75, 'load_saheart', 1.29, 1.72, 462, 9, 9, 160, 'saheart', 'keel'],
       [1.25, 0.72, 'load_australian', 1.35, 1.88, 690, 18, 14, 307, 'australian', 'keel'],
       [1.21, 0.72, 'load_crx', 1.62, 2.26, 653, 29, 15, 296, 'crx', 'keel'],
       [36.0, 0.57, 'load_lymphography', 4.45, 7.83, 148, 38, 18, 4, 'lymphography', 'keel'],
       [1.68, 0.72, 'load_wdbc', 2.11, 2.93, 569, 30, 30, 212, 'wdbc', 'keel'],
       [1.79, 0.22, 'load_ionosphere', 1.23, 5.51, 351, 33, 33, 126, 'ionosphere', 'keel'],
       [3.85, 1.39, 'load_spectfheart', 4.9, 3.52, 267, 44, 44, 55, 'spectfheart', 'keel']], dtype=object)