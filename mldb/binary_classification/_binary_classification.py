import numpy as np
import pandas as pd

from mldb._io import read_csv_data, read_arff_data, construct_return_set, citations

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
            'get_data_loaders',
            'get_filtered_data_loaders']

def load_hiva(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/hiva/hiva_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/classification/hiva/hiva_train.labels')
    db['target']= target
    
    return construct_return_set(db, "HIVA", return_X_y, encode, citation= citations['krnn'], name= "HIVA", verbose= verbose)

def load_hypothyroid(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/hypothyroid/hypothyroid.data.txt')
    db.columns= ['target'] + list(db.columns[1:])
    
    return construct_return_set(db, "hypothyroid", return_X_y, encode, citation= citations['krnn'], name= "hypothyroid", verbose= verbose)

def load_sylva(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/sylva/sylva_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/classification/sylva/sylva_train.labels')
    db['target']= target
    
    return construct_return_set(db, "sylva", return_X_y, encode, citation= citations['krnn'], name= "sylva", verbose= verbose)

def load_pc1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/pc1/pc1.arff')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "PC1", return_X_y, encode, citation= citations['krnn'], name= "PC1", verbose= verbose)

def load_cm1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/cm1/cm1.arff.txt')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "CM1", return_X_y, encode, citation= citations['krnn'], name= "CM1", verbose= verbose)

def load_kc1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kc1/kc1.arff.txt')
    db= pd.DataFrame(data)
    db.loc[db['defects'] == b'false', 'defects']= False
    db.loc[db['defects'] == b'true', 'defects']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "KC1", return_X_y, encode, citation= citations['krnn'], name= "KC1", verbose= verbose)

def load_spectf(return_X_y= False, encode= True, verbose= False):
    db0= read_csv_data('data/classification/spect_f/SPECTF.train.txt')
    db1= read_csv_data('data/classification/spect_f/SPECTF.test.txt')
    db= pd.concat([db0, db1])
    db.columns= ['target'] + list(db.columns[1:])
    
    return construct_return_set(db, "SPECT_F", return_X_y, encode, citation= citations['krnn'], name= "SPECT_F", verbose= verbose)

def load_hepatitis(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/hepatitis/hepatitis.data.txt')
    db.columns= ['target'] + list(db.columns[1:])

    return construct_return_set(db, "hepatitis", return_X_y, encode, citation= citations['krnn'], name= "hepatitis", verbose= verbose)

def load_vehicle(return_X_y= False, encode= True, verbose= False):
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
    
    return construct_return_set(db, "vehicle", return_X_y, encode, citation= citations['krnn'], name= "vehicle", verbose= verbose)

def load_ada(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/ada/ada_train.data', sep= ' ')
    del db[db.columns[-1]]
    target= read_csv_data('data/classification/ada/ada_train.labels')
    db['target']= target
    
    return construct_return_set(db, "ADA", return_X_y, encode, citation= citations['krnn'], name= "ADA", verbose= verbose)

def load_german(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/german/german.data.txt', sep= ' ')
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "german", return_X_y, encode, encoding_threshold= 20, citation= citations['krnn'], name= "german", verbose= verbose)

def load_glass(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    db.loc[db['target'] != 3, 'target']= 0
    del db[db.columns[0]]
    
    return construct_return_set(db, "glass", return_X_y, encode, citation= citations['krnn'], name= "glass", verbose= verbose)

def load_satimage(return_X_y= False, encode= True, verbose= False):
    db0= read_csv_data('data/classification/satimage/sat.trn.txt', sep= ' ')
    db1= read_csv_data('data/classification/satimage/sat.tst.txt', sep= ' ')
    db= pd.concat([db0, db1])
    db.columns= list(db.columns[:-1]) + ['target']
    db.loc[db['target'] != 4, 'target']= 0
    
    return construct_return_set(db, "SATIMAGE", return_X_y, encode, citation= citations['krnn'], name= "SATIMAGE", verbose= verbose)

def load_abalone_17_vs_7_8_9_10(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/abalone-17_vs_7-8-9-10/abalone-17_vs_7-8-9-10.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone_17_vs_7_8_9_10", return_X_y, encode, citation= citations['keel'], name= "abalone_17_vs_7_8_9_10", verbose= verbose)

def load_abalone_19_vs_10_11_12_13(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/abalone-19_vs_10-11-12-13/abalone-19_vs_10-11-12-13.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-19_vs_10-11-12-13", return_X_y, encode, citation= citations['keel'], name= "abalone-19_vs_10-11-12-13", verbose= verbose)

def load_abalone_20_vs_8_9_10(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/abalone-20_vs_8-9-10/abalone-20_vs_8-9-10.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-20_vs_8-9-10", return_X_y, encode, citation= citations['keel'], name= "abalone-20_vs_8-9-10", verbose= verbose)

def load_abalone_21_vs_8(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/abalone-21_vs_8/abalone-21_vs_8.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-21_vs_8", return_X_y, encode, citation= citations['keel'], name= "abalone-21_vs_8", verbose= verbose)

def load_abalone_3_vs_11(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/abalone-3_vs_11/abalone-3_vs_11.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone-3_vs_11", return_X_y, encode, citation= citations['keel'], name= "abalone-3_vs_11", verbose= verbose)

def load_abalone19(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/abalone19/abalone19.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone19", return_X_y, encode, citation= citations['keel'], name= "abalone19", verbose= verbose)

def load_abalone9_18(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/abalone9-18/abalone9-18.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "abalone9-18", return_X_y, encode, citation= citations['keel'], name= "abalone9-18", verbose= verbose)

def load_car_good(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/car-good/car-good.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "car_good", return_X_y, encode, citation= citations['keel'], name= "car_good", verbose= verbose)

def load_car_vgood(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/car-vgood/car-vgood.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "car-vgood", return_X_y, encode, citation= citations['keel'], name= "car-vgood", verbose= verbose)

def load_cleveland_0_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/cleveland-0_vs_4/cleveland-0_vs_4_no_null.dat')
    db= pd.DataFrame(data)
    db.loc[db['num'] == b'negative', 'num']= False
    db.loc[db['num'] == b'positive', 'num']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "cleveland-0_vs_4", return_X_y, encode, citation= citations['keel'], name= "cleveland-0_vs_4", verbose= verbose)

def load_dermatology_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/dermatology-6/dermatology-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "dermatology-6", return_X_y, encode, citation= citations['keel'], name= "dermatology-6", verbose= verbose)

def load_ecoli_0_1_3_7_vs_2_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-1-3-7_vs_2-6/ecoli-0-1-3-7_vs_2-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-3-7_vs_2-6", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-1-3-7_vs_2-6", verbose= verbose)

def load_ecoli_0_1_4_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-1-4-6_vs_5/ecoli-0-1-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-6_vs_5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-1-4-6_vs_5", verbose= verbose)

def load_ecoli_0_1_4_7_vs_2_3_5_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-1-4-7_vs_2-3-5-6/ecoli-0-1-4-7_vs_2-3-5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-7_vs_2-3-5-6", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-1-4-7_vs_2-3-5-6", verbose= verbose)

def load_ecoli_0_1_4_7_vs_5_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-1-4-7_vs_5-6/ecoli-0-1-4-7_vs_5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1-4-7_vs_5-6", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-1-4-7_vs_5-6", verbose= verbose)

def load_ecoli_0_1_vs_2_3_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-1_vs_2-3-5/ecoli-0-1_vs_2-3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1_vs_2-3-5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-1_vs_2-3-5", verbose= verbose)

def load_ecoli_0_1_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-1_vs_5/ecoli-0-1_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-1_vs_5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-1_vs_5", verbose= verbose)

def load_ecoli_0_2_3_4_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-2-3-4_vs_5/ecoli-0-2-3-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-2-3-4_vs_5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-2-3-4_vs_5", verbose= verbose)

def load_ecoli_0_2_6_7_vs_3_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-2-6-7_vs_3-5/ecoli-0-2-6-7_vs_3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-2-6-7_vs_3-5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-2-6-7_vs_3-5", verbose= verbose)

def load_ecoli_0_3_4_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-3-4-6_vs_5/ecoli-0-3-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4-6_vs_5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-3-4-6_vs_5", verbose= verbose)

def load_ecoli_0_3_4_7_vs_5_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-3-4-7_vs_5-6/ecoli-0-3-4-7_vs_5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4-7_vs_5-6", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-3-4-7_vs_5-6", verbose= verbose)

def load_ecoli_0_3_4_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-3-4_vs_5/ecoli-0-3-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-3-4_vs_5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-3-4_vs_5", verbose= verbose)

def load_ecoli_0_4_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-4-6_vs_5/ecoli-0-4-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-4-6_vs_5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-4-6_vs_5", verbose= verbose)

def load_ecoli_0_6_7_vs_3_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-6-7_vs_3-5/ecoli-0-6-7_vs_3-5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-6-7_vs_3-5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-6-7_vs_3-5", verbose= verbose)

def load_ecoli_0_6_7_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0-6-7_vs_5/ecoli-0-6-7_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0-6-7_vs_5", return_X_y, encode, citation= citations['keel'], name= "ecoli-0-6-7_vs_5", verbose= verbose)

def load_ecoli4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli4/ecoli4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli4", return_X_y, encode, citation= citations['keel'], name= "ecoli4", verbose= verbose)

def load_flaref(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/flare-F/flare-F.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "flare-F", return_X_y, encode, citation= citations['keel'], name= "flare-F", verbose= verbose)

def load_glass_0_1_4_6_vs_2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass-0-1-4-6_vs_2/glass-0-1-4-6_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-4-6_vs_2", return_X_y, encode, citation= citations['keel'], name= "glass-0-1-4-6_vs_2", verbose= verbose)

def load_glass_0_1_5_vs_2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass-0-1-5_vs_2/glass-0-1-5_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-5_vs_2", return_X_y, encode, citation= citations['keel'], name= "glass-0-1-5_vs_2", verbose= verbose)

def load_glass_0_1_6_vs_2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass-0-1-6_vs_2/glass-0-1-6_vs_2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-6_vs_2", return_X_y, encode, citation= citations['keel'], name= "glass-0-1-6_vs_2", verbose= verbose)

def load_glass_0_1_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass-0-1-6_vs_5/glass-0-1-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-6_vs_5", return_X_y, encode, citation= citations['keel'], name= "glass-0-1-6_vs_5", verbose= verbose)

def load_glass_0_4_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass-0-4_vs_5/glass-0-4_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-4_vs_5", return_X_y, encode, citation= citations['keel'], name= "glass-0-4_vs_5", verbose= verbose)

def load_glass_0_6_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass-0-6_vs_5/glass-0-6_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['typeGlass'] == b'negative', 'typeGlass']= False
    db.loc[db['typeGlass'] == b'positive', 'typeGlass']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-6_vs_5", return_X_y, encode, citation= citations['keel'], name= "glass-0-6_vs_5", verbose= verbose)

def load_glass2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass2/glass2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass2", return_X_y, encode, citation= citations['keel'], name= "glass2", verbose= verbose)

def load_glass4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass4/glass4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass4", return_X_y, encode, citation= citations['keel'], name= "glass4", verbose= verbose)

def load_glass5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass5/glass5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass5", return_X_y, encode, citation= citations['keel'], name= "glass5", verbose= verbose)

def load_kddcup_buffer_overflow_vs_back(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kddcup-buffer_overflow_vs_back/kddcup-buffer_overflow_vs_back.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-buffer_overflow_vs_back", return_X_y, encode, citation= citations['keel'], name= "kddcup-buffer_overflow_vs_back", verbose= verbose)

def load_kddcup_guess_passwd_vs_satan(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kddcup-guess_passwd_vs_satan/kddcup-guess_passwd_vs_satan.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-guess_passwd_vs_satan", return_X_y, encode, citation= citations['keel'], name= "kddcup-guess_passwd_vs_satan", verbose= verbose)

def load_kddcup_land_vs_portsweep(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kddcup-land_vs_portsweep/kddcup-land_vs_portsweep.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-land_vs_portsweep", return_X_y, encode, citation= citations['keel'], name= "kddcup-land_vs_portsweep", verbose= verbose)

def load_kddcup_land_vs_satan(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kddcup-land_vs_satan/kddcup-land_vs_satan.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-land_vs_satan", return_X_y, encode, citation= citations['keel'], name= "kddcup-land_vs_satan", verbose= verbose)

def load_kddcup_rootkit_imap_vs_back(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kddcup-rootkit-imap_vs_back/kddcup-rootkit-imap_vs_back.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kddcup-rootkit-imap_vs_back", return_X_y, encode, citation= citations['keel'], name= "kddcup-rootkit-imap_vs_back", verbose= verbose)

def load_kr_vs_k_one_vs_fifteen(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kr-vs-k-one_vs_fifteen/kr-vs-k-one_vs_fifteen.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-one_vs_fifteen", return_X_y, encode, citation= citations['keel'], name= "kr-vs-k-one_vs_fifteen", verbose= verbose)

def load_kr_vs_k_three_vs_eleven(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kr-vs-k-three_vs_eleven/kr-vs-k-three_vs_eleven.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-three_vs_eleven", return_X_y, encode, citation= citations['keel'], name= "kr-vs-k-three_vs_eleven", verbose= verbose)

def load_kr_vs_k_zero_one_vs_draw(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kr-vs-k-zero-one_vs_draw/kr-vs-k-zero-one_vs_draw.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero-one_vs_draw", return_X_y, encode, citation= citations['keel'], name= "kr-vs-k-zero-one_vs_draw", verbose= verbose)

def load_kr_vs_k_zero_vs_eight(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kr-vs-k-zero_vs_eight/kr-vs-k-zero_vs_eight.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero_vs_eight", return_X_y, encode, citation= citations['keel'], name= "kr-vs-k-zero_vs_eight", verbose= verbose)

def load_kr_vs_k_zero_vs_fifteen(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kr-vs-k-zero_vs_fifteen/kr-vs-k-zero_vs_fifteen.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr-vs-k-zero_vs_fifteen", return_X_y, encode, citation= citations['keel'], name= "kr-vs-k-zero_vs_fifteen", verbose= verbose)

def load_led7digit_0_2_4_5_6_7_8_9_vs_1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/led7digit-0-2-4-5-6-7-8-9_vs_1/led7digit-0-2-4-5-6-7-8-9_vs_1.dat')
    db= pd.DataFrame(data)
    db.loc[db['number'] == b'negative', 'number']= False
    db.loc[db['number'] == b'positive', 'number']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "led7digit-0-2-4-6-7-8-9_vs_1", return_X_y, encode, citation= citations['keel'], name= "led7digit-0-2-4-6-7-8-9_vs_1", verbose= verbose)

def load_lymphography_normal_fibrosis(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/lymphography-normal-fibrosis/lymphography-normal-fibrosis.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "lymphography-normal-fibrosis", return_X_y, encode, citation= citations['keel'], name= "lymphography-normal-fibrosis", verbose= verbose)

def load_page_blocks_1_3_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/page-blocks-1-3_vs_4/page-blocks-1-3_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "page-blocks-1-3_vs_4", return_X_y, encode, citation= citations['keel'], name= "page-blocks-1-3_vs_4", verbose= verbose)

def load_poker_8_9_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/poker-8-9_vs_5/poker-8-9_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8-9_vs_5", return_X_y, encode, citation= citations['keel'], name= "poker-8-9_vs_5", verbose= verbose)

def load_poker_8_9_vs_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/poker-8-9_vs_6/poker-8-9_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8-9_vs_6", return_X_y, encode, citation= citations['keel'], name= "poker-8-9_vs_6", verbose= verbose)

def load_poker_8_vs_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/poker-8_vs_6/poker-8_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-8_vs_6", return_X_y, encode, citation= citations['keel'], name= "poker-8_vs_6", verbose= verbose)

def load_poker_9_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/poker-9_vs_7/poker-9_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "poker-9_vs_7", return_X_y, encode, citation= citations['keel'], name= "poker-9_vs_7", verbose= verbose)

def load_shuttle_2_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/shuttle-2_vs_5/shuttle-2_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-2_vs_5", return_X_y, encode, citation= citations['keel'], name= "shuttle-2_vs_5", verbose= verbose)

def load_shuttle_6_vs_2_3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/shuttle-6_vs_2-3/shuttle-6_vs_2-3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-6_vs_2-3", return_X_y, encode, citation= citations['keel'], name= "shuttle-6_vs_2-3", verbose= verbose)

def load_shuttle_c0_vs_c4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/shuttle-c0-vs-c4/shuttle-c0-vs-c4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-c0-vs-c4", return_X_y, encode, citation= citations['keel'], name= "shuttle-c0-vs-c4", verbose= verbose)

def load_shuttle_c2_vs_c4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/shuttle-c2-vs-c4/shuttle-c2-vs-c4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "shuttle-c2-vs-c4", return_X_y, encode, citation= citations['keel'], name= "shuttle-c2-vs-c4", verbose= verbose)

def load_vowel0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/vowel0/vowel0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vowel0", return_X_y, encode, citation= citations['keel'], name= "vowel0", verbose= verbose)

def load_winequality_red_3_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/winequality-red-3_vs_5/winequality-red-3_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-3_vs_5", return_X_y, encode, citation= citations['keel'], name= "winequality-red-3_vs_5", verbose= verbose)

def load_winequality_red_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/winequality-red-4/winequality-red-4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-4", return_X_y, encode, citation= citations['keel'], name= "winequality-red-4", verbose= verbose)

def load_winequality_red_8_vs_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/winequality-red-8_vs_6/winequality-red-8_vs_6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-8_vs_6", return_X_y, encode, citation= citations['keel'], name= "winequality-red-8_vs_6", verbose= verbose)

def load_winequality_red_8_vs_6_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/winequality-red-8_vs_6-7/winequality-red-8_vs_6-7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-red-8_vs_6-7", return_X_y, encode, citation= citations['keel'], name= "winequality-red-8_vs_6-7", verbose= verbose)

def load_winequality_white_3_9_vs_5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/winequality-white-3-9_vs_5/winequality-white-3-9_vs_5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-3-9_vs_5", return_X_y, encode, citation= citations['keel'], name= "winequality-white-3-9_vs_5", verbose= verbose)

def load_winequality_white_3_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/winequality-white-3_vs_7/winequality-white-3_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-3_vs_7", return_X_y, encode, citation= citations['keel'], name= "winequality-white-3_vs_7", verbose= verbose)

def load_winequality_white_9_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/winequality-white-9_vs_4/winequality-white-9_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "winequality-white-9_vs_4", return_X_y, encode, citation= citations['keel'], name= "winequality-white-9_vs_4", verbose= verbose)

def load_yeast_0_2_5_6_vs_3_7_8_9(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-0-2-5-6_vs_3-7-8-9/yeast-0-2-5-6_vs_3-7-8-9.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-2-5-6_vs_3-7-8-9", return_X_y, encode, citation= citations['keel'], name= "yeast-0-2-5-6_vs_3-7-8-9", verbose= verbose)

def load_yeast_0_2_5_7_9_vs_3_6_8(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-0-2-5-7-9_vs_3-6-8/yeast-0-2-5-7-9_vs_3-6-8.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-2-5-7-9_vs_3-6-8", return_X_y, encode, citation= citations['keel'], name= "yeast-0-2-5-7-9_vs_3-6-8", verbose= verbose)

def load_yeast_0_3_5_9_vs_7_8(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-0-3-5-9_vs_7-8/yeast-0-3-5-9_vs_7-8.dat')
    db= pd.DataFrame(data)
    db.loc[db['class'] == b'negative', 'class']= False
    db.loc[db['class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-3-5-9_vs_7-8", return_X_y, encode, citation= citations['keel'], name= "yeast-0-3-5-9_vs_7-8", verbose= verbose)

def load_yeast_0_5_6_7_9_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-0-5-6-7-9_vs_4/yeast-0-5-6-7-9_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-0-5-6-7-9_vs_4", return_X_y, encode, citation= citations['keel'], name= "yeast-0-5-6-7-9_vs_4", verbose= verbose)

def load_yeast_1_2_8_9_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-1-2-8-9_vs_7/yeast-1-2-8-9_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1-2-8-9_vs_7", return_X_y, encode, citation= citations['keel'], name= "yeast-1-2-8-9_vs_7", verbose= verbose)

def load_yeast_1_4_5_8_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-1-4-5-8_vs_7/yeast-1-4-5-8_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1-4-5-8_vs_7", return_X_y, encode, citation= citations['keel'], name= "yeast-1-4-5-8_vs_7", verbose= verbose)

def load_yeast_1_vs_7(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-1_vs_7/yeast-1_vs_7.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-1_vs_7", return_X_y, encode, citation= citations['keel'], name= "yeast-1_vs_7", verbose= verbose)

def load_yeast_2_vs_4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-2_vs_4/yeast-2_vs_4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-2_vs_4", return_X_y, encode, citation= citations['keel'], name= "yeast-2_vs_4", verbose= verbose)

def load_yeast_2_vs_8(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast-2_vs_8/yeast-2_vs_8.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast-2_vs_8", return_X_y, encode, citation= citations['keel'], name= "yeast-2_vs_8", verbose= verbose)

def load_yeast4(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast4/yeast4.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast4", return_X_y, encode, citation= citations['keel'], name= "yeast4", verbose= verbose)

def load_yeast5(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast5/yeast5.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast5", return_X_y, encode, citation= citations['keel'], name= "yeast5", verbose= verbose)

def load_yeast6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast6/yeast6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast6", return_X_y, encode, citation= citations['keel'], name= "yeast6", verbose= verbose)

def load_zoo_3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/zoo-3/zoo-3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "zoo-3", return_X_y, encode, citation= citations['keel'], name= "zoo-3", verbose= verbose)

#########################

def load_ecoli_0_vs_1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli-0_vs_1/ecoli-0_vs_1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'class']= False
    db.loc[db['Class'] == b'positive', 'class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli-0_vs_1", return_X_y, encode, citation= citations['keel'], name= "ecoli-0_vs_1", verbose= verbose)

def load_ecoli1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli1/ecoli1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli1", return_X_y, encode, citation= citations['keel'], name= "ecoli1", verbose= verbose)

def load_ecoli2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli2/ecoli2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli2", return_X_y, encode, citation= citations['keel'], name= "ecoli2", verbose= verbose)

def load_ecoli3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ecoli3/ecoli3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ecoli3", return_X_y, encode, citation= citations['keel'], name= "ecoli3", verbose= verbose)

def load_glass_0_1_2_3_vs_4_5_6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass-0-1-2-3_vs_4-5-6/glass-0-1-2-3_vs_4-5-6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass-0-1-2-3_vs_4-5-6", return_X_y, encode, citation= citations['keel'], name= "glass-0-1-2-3_vs_4-5-6", verbose= verbose)

def load_glass0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass0/glass0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass0", return_X_y, encode, citation= citations['keel'], name= "glass0", verbose= verbose)

def load_glass1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass1/glass1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass1", return_X_y, encode, citation= citations['keel'], name= "glass1", verbose= verbose)

def load_glass6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/glass6/glass6.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "glass6", return_X_y, encode, citation= citations['keel'], name= "glass6", verbose= verbose)

def load_habarman(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/haberman/haberman.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "habarman", return_X_y, encode, citation= citations['keel'], name= "habarman", verbose= verbose)

def load_iris0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/iris0/iris0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "iris0", return_X_y, encode, citation= citations['keel'], name= "iris0", verbose= verbose)

def load_new_thyroid1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/new_thyroid1/new-thyroid1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "new_thyroid1", return_X_y, encode, citation= citations['keel'], name= "new_thyroid1", verbose= verbose)

def load_new_thyroid2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/new_thyroid2/new_thyroid2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "new_thyroid2", return_X_y, encode, citation= citations['keel'], name= "new_thyroid2", verbose= verbose)

def load_page_blocks0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/page-blocks0/page-blocks0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "page_blocks0", return_X_y, encode, citation= citations['keel'], name= "page_blocks0", verbose= verbose)

def load_pima(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/pima/pima.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "pima", return_X_y, encode, citation= citations['keel'], name= "pima", verbose= verbose)

def load_segment0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/segment0/segment0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "segment0", return_X_y, encode, citation= citations['keel'], name= "segment0", verbose= verbose)

def load_vehicle0(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/vehicle0/vehicle0.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle0", return_X_y, encode, citation= citations['keel'], name= "vehicle0", verbose= verbose)

def load_vehicle1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/vehicle1/vehicle1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle1", return_X_y, encode, citation= citations['keel'], name= "vehicle1", verbose= verbose)

def load_vehicle2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/vehicle2/vehicle2.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle2", return_X_y, encode, citation= citations['keel'], name= "vehicle2", verbose= verbose)

def load_vehicle3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/vehicle3/vehicle3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vehicle3", return_X_y, encode, citation= citations['keel'], name= "vehicle3", verbose= verbose)

def load_wisconsin(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/wisconsin/wisconsin.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wisconsin", return_X_y, encode, citation= citations['keel'], name= "wisconsin", verbose= verbose)

def load_yeast1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast1/yeast1.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast1", return_X_y, encode, citation= citations['keel'], name= "yeast1", verbose= verbose)

def load_yeast3(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/yeast3/yeast3.dat')
    db= pd.DataFrame(data)
    db.loc[db['Class'] == b'negative', 'Class']= False
    db.loc[db['Class'] == b'positive', 'Class']= True
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "yeast3", return_X_y, encode, citation= citations['keel'], name= "yeast3", verbose= verbose)

def load_mammographic(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/mammographic/mammographic.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "mammographic", return_X_y, encode, citation= citations['keel'], name='mammographic', verbose=verbose)

def load_bupa(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/bupa/bupa.dat')
    db= pd.DataFrame(data)
    db.loc[db[db.columns[-1]] == 1, db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == 2, db.columns[-1]]= 1
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "bupa", return_X_y, encode, citation= citations['keel'], name="bupa", verbose=verbose)

def load_monk_2(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/monk-2/monk-2.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "monk-2", return_X_y, encode, citation= citations['keel'], name="monk-2", verbose=verbose)

def load_appendicitis(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/appendicitis/appendicitis.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "appendicitis", return_X_y, encode, citation= citations['keel'], name="appendicitis", verbose=verbose)

def load_saheart(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/saheart/saheart.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "saheart", return_X_y, encode, citation= citations['keel'], name="saheart", verbose=verbose)

def load_australian(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/australian/australian.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "australian", return_X_y, encode, citation= citations['keel'], name="australian", verbose=verbose)

def load_crx(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/crx/crx.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "crx", return_X_y, encode, citation= citations['keel'], name="crx", verbose=verbose)

def load_lymphography(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/lymphography/lymphography.dat')
    db= pd.DataFrame(data)
    db.loc[db[db.columns[-1]] == 'metastases', db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == 'malign_lymph', db.columns[-1]]= 1
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "lymphography", return_X_y, encode, citation= citations['keel'], name="lymphography", verbose=verbose)

def load_wdbc(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/wdbc/wdbc.dat')
    db= pd.DataFrame(data)
    db.loc[db[db.columns[-1]] == 'M', db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == 'B', db.columns[-1]]= 1
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wdbc", return_X_y, encode, citation= citations['keel'], name="wdbc", verbose=verbose)

def load_ionosphere(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/ionosphere/ionosphere.dat')
    db= pd.DataFrame(data)
    db.loc[db[db.columns[-1]] == 'g', db.columns[-1]]= 0
    db.loc[db[db.columns[-1]] == 'b', db.columns[-1]]= 1
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ionosphere", return_X_y, encode, citation= citations['keel'], name="ionosphere", verbose=verbose)

def load_spectfheart(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/spectfheart/spectfheart.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "spectfheart", return_X_y, encode, citation= citations['keel'], name="spectfheart", verbose=verbose)

def summary(include_citation= True, ratio_dist= False, subset= 'study'):
    results= []
    # fixing the globals dictionary keys
    
    for func_name in __all__:
        if func_name.startswith('load_'):
            data_not_encoded= globals()[func_name](return_X_y= False, encode= False)
            data_encoded= globals()[func_name](return_X_y= False, encode= True)
            
            X= data_encoded['data']
            
            X_min= X[data_encoded['target'] == 1]
            X_maj= X[data_encoded['target'] == 0]
            
            result= {'loader_function': globals()[func_name],
                            'name': data_not_encoded['name'],
                            'len': len(data_not_encoded['data']),
                            'non_encoded_n_attr': len(data_not_encoded['data'][0]),
                            'encoded_n_attr': len(data_encoded['data'][0]),
                            'n_minority': np.sum(data_encoded['target'] == 1),
                            'imbalance_ratio': np.sum(data_encoded['target'] == 0)/np.sum(data_encoded['target'] == 1)}
            
            if ratio_dist:
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
            
            if include_citation:
                result['citation']= data_encoded['citation']
            
            results.append(result)
    
    df_results= pd.DataFrame(results)

    return df_results

def get_filtered_data_loaders(num_features_lower_bound= 1,
                              num_features_upper_bound= 5000,
                              len_lower_bound= 1,
                              len_upper_bound= 10000,
                              imbalance_ratio_lower_bound= 0,
                              imbalance_ratio_upper_bound= 1e10):
    descriptors= summary(ratio_dist= False)
    return descriptors[(descriptors['len'] >= len_lower_bound) & 
                       (descriptors['len'] < len_upper_bound) & 
                       (descriptors['encoded_n_attr'] >= num_features_lower_bound) & 
                       (descriptors['encoded_n_attr'] < num_features_upper_bound) & 
                       (descriptors['imbalance_ratio'] >= imbalance_ratio_lower_bound) & 
                       (descriptors['imbalance_ratio'] < imbalance_ratio_upper_bound)]['loader_function'].values

def get_data_loaders(subset='all'):
    """
    Args:
        subset (str): 'all'/'study'/'small'/'tiny'
    """
    
    num_features_lower_bound= 1
    len_lower_bound= 1
    imbalance_ratio_lower_bound= 0
    num_features_upper_bound= 1e10
    len_upper_bound= 1e10
    imbalance_ratio_upper_bound= 1e10
    
    if subset == 'study':
        num_features_upper_bound= 100
        len_upper_bound= 4000
    elif subset == 'small':
        num_features_upper_bound= 100
        len_upper_bound= 1000
    elif subset == 'tiny':
        len_upper_bound= 120
    
    return get_filtered_data_loaders(num_features_lower_bound,
                                     num_features_upper_bound,
                                     len_lower_bound,
                                     len_upper_bound,
                                     imbalance_ratio_lower_bound,
                                     imbalance_ratio_upper_bound)
