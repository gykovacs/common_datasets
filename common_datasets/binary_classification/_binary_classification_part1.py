"""
This module contains the binary classification data loaders
"""

import pandas as pd
import numpy as np

from .._io import (read_csv_data, read_arff_data,
                        load_arff_template_binary,
                        prepare_csv_data_template,
                        DataPreprocessor)

__all__= ['load_kddcup_buffer_overflow_vs_back',
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
            'load_ada'
#            'load_hiva'
             ]

def load_kddcup_buffer_overflow_vs_back():
    """
    Load the kddcup_buffer_overflow_vs_back dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kddcup-buffer_overflow_vs_back/kddcup-buffer_overflow_vs_back.dat'
    return load_arff_template_binary(path=path,
                                        name="kddcup-buffer_overflow_vs_back",
                                        target_label='Class')

def load_kddcup_guess_passwd_vs_satan():
    """
    Load the kddcup-guess_passwd_vs_satan dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kddcup-guess_passwd_vs_satan/kddcup-guess_passwd_vs_satan.dat'
    return load_arff_template_binary(path=path,
                                        name="kddcup-guess_passwd_vs_satan",
                                        target_label='Class')

def load_kddcup_land_vs_portsweep():
    """
    Load the kddcup-land_vs_portsweep dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kddcup-land_vs_portsweep/kddcup-land_vs_portsweep.dat'
    return load_arff_template_binary(path=path,
                                        name="kddcup-land_vs_portsweep",
                                        target_label='Class')

def load_kddcup_land_vs_satan():
    """
    Load the kddcup-land_vs_satan dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kddcup-land_vs_satan/kddcup-land_vs_satan.dat'
    return load_arff_template_binary(path=path,
                                        name="kddcup-land_vs_satan",
                                        target_label='Class')

def load_kddcup_rootkit_imap_vs_back():
    """
    Load the kddcup-rootkit-imap_vs_back dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kddcup-rootkit-imap_vs_back/kddcup-rootkit-imap_vs_back.dat'
    return load_arff_template_binary(path=path,
                                        name="kddcup-rootkit-imap_vs_back",
                                        target_label='Class')

def load_kr_vs_k_one_vs_fifteen():
    """
    Load the kr_vs_k_one_vs_fifteen dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kr-vs-k-one_vs_fifteen/kr-vs-k-one_vs_fifteen.dat'
    return load_arff_template_binary(path=path,
                                        name="kr_vs_k_one_vs_fifteen",
                                        target_label='Class')

def load_kr_vs_k_three_vs_eleven():
    """
    Load the kr-vs-k-three_vs_eleven dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kr-vs-k-three_vs_eleven/kr-vs-k-three_vs_eleven.dat'
    return load_arff_template_binary(path=path,
                                        name="kr-vs-k-three_vs_eleven",
                                        target_label='Class')

def load_kr_vs_k_zero_one_vs_draw():
    """
    Load the kr-vs-k-zero-one_vs_draw dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kr-vs-k-zero-one_vs_draw/kr-vs-k-zero-one_vs_draw.dat'
    return load_arff_template_binary(path=path,
                                        name="kr-vs-k-zero-one_vs_draw",
                                        target_label='Class')

def load_kr_vs_k_zero_vs_eight():
    """
    Load the kr-vs-k-zero_vs_eight dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kr-vs-k-zero_vs_eight/kr-vs-k-zero_vs_eight.dat'
    return load_arff_template_binary(path=path,
                                        name="kr-vs-k-zero_vs_eight",
                                        target_label='Class')

def load_kr_vs_k_zero_vs_fifteen():
    """
    Load the kr-vs-k-zero_vs_fifteen dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kr-vs-k-zero_vs_fifteen/kr-vs-k-zero_vs_fifteen.dat'
    return load_arff_template_binary(path=path,
                                        name="kr-vs-k-zero_vs_fifteen",
                                        target_label='Class')

def load_poker_8_9_vs_5():
    """
    Load the poker-8-9_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/poker-8-9_vs_5/poker-8-9_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name="poker-8-9_vs_5",
                                        target_label='Class')

def load_poker_8_9_vs_6():
    """
    Load the poker-8-9_vs_6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/poker-8-9_vs_6/poker-8-9_vs_6.dat'
    return load_arff_template_binary(path=path,
                                        name="poker-8-9_vs_6",
                                        target_label='Class')

def load_poker_8_vs_6():
    """
    Load the poker-8_vs_6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/poker-8_vs_6/poker-8_vs_6.dat'
    return load_arff_template_binary(path=path,
                                        name="poker-8_vs_6",
                                        target_label='Class')

def load_poker_9_vs_7():
    """
    Load the poker-9_vs_7 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/poker-9_vs_7/poker-9_vs_7.dat'
    return load_arff_template_binary(path=path,
                                        name="poker-9_vs_7",
                                        target_label='Class')

def load_shuttle_2_vs_5():
    """
    Load the shuttle-2_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/shuttle-2_vs_5/shuttle-2_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name="shuttle-2_vs_5",
                                        target_label='Class')

def load_shuttle_6_vs_2_3():
    """
    Load the shuttle-6_vs_2-3 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/shuttle-6_vs_2-3/shuttle-6_vs_2-3.dat'
    return load_arff_template_binary(path=path,
                                        name="shuttle-6_vs_2-3",
                                        target_label='Class')

def load_shuttle_c0_vs_c4():
    """
    Load the shuttle-c0-vs-c4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/shuttle-c0-vs-c4/shuttle-c0-vs-c4.dat'
    return load_arff_template_binary(path=path,
                                        name="shuttle-c0-vs-c4",
                                        target_label='Class')

def load_shuttle_c2_vs_c4():
    """
    Load the shuttle-c2-vs-c4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/shuttle-c2-vs-c4/shuttle-c2-vs-c4.dat'
    return load_arff_template_binary(path=path,
                                        name="shuttle-c2-vs-c4",
                                        target_label='Class')

def load_vehicle0():
    """
    Load the vehicle0 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/vehicle0/vehicle0.dat'
    return load_arff_template_binary(path=path,
                                        name="vehicle0",
                                        target_label='Class')

def load_vehicle1():
    """
    Load the vehicle1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/vehicle1/vehicle1.dat'
    return load_arff_template_binary(path=path,
                                        name="vehicle1",
                                        target_label='Class')

def load_vehicle2():
    """
    Load the vehicle2 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/vehicle2/vehicle2.dat'
    return load_arff_template_binary(path=path,
                                        name="vehicle2",
                                        target_label='Class')

def load_vehicle3():
    """
    Load the vehicle3 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/vehicle3/vehicle3.dat'
    return load_arff_template_binary(path=path,
                              name="vehicle3",
                              target_label='Class')

def load_pc1():
    """
    Load the PC1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/pc1/pc1.arff'
    return load_arff_template_binary(path=path,
                                        name="PC1",
                                        target_label='defects',
                                        citation_key='krnn')

def load_cm1():
    """
    Load the CM1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/cm1/cm1.arff.txt'
    return load_arff_template_binary(path=path,
                                        name="CM1",
                                        target_label='defects',
                                        citation_key='krnn')

def load_kc1():
    """
    Load the KC1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/kc1/kc1.arff.txt'
    return load_arff_template_binary(path=path,
                                        name="KC1",
                                        target_label='defects',
                                        citation_key='krnn')

def load_car_good():
    """
    Load the car_good dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/car-good/car-good.dat'
    return load_arff_template_binary(path=path,
                                        name="car_good",
                                        target_label='Class')

def load_car_vgood():
    """
    Load the car-vgood dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/car-vgood/car-vgood.dat'
    return load_arff_template_binary(path=path,
                                        name="car-vgood",
                                        target_label='Class')

def load_cleveland_0_vs_4():
    """
    Load the cleveland-0_vs_4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/cleveland-0_vs_4/cleveland-0_vs_4_no_null.dat'
    return load_arff_template_binary(path=path,
                                        name="cleveland-0_vs_4",
                                        target_label='num')

def load_dermatology_6():
    """
    Load the dermatology-6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/dermatology-6/dermatology-6.dat'
    return load_arff_template_binary(path=path,
                                        name="dermatology-6",
                                        target_label='Class')

def load_flaref():
    """
    Load the flare-F dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/flare-F/flare-F.dat'
    return load_arff_template_binary(path=path,
                                        name="flare-F",
                                        target_label='Class')

def load_led7digit_0_2_4_5_6_7_8_9_vs_1():
    """
    Load the led7digit-0-2-4-6-7-8-9_vs_1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/led7digit-0-2-4-5-6-7-8-9_vs_1/led7digit-0-2-4-5-6-7-8-9_vs_1.dat'
    return load_arff_template_binary(path=path,
                                        name="led7digit-0-2-4-6-7-8-9_vs_1",
                                        target_label='number')

def load_lymphography_normal_fibrosis():
    """
    Load the lymphography-normal-fibrosis dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/lymphography-normal-fibrosis/lymphography-normal-fibrosis.dat'
    return load_arff_template_binary(path=path,
                                        name="lymphography-normal-fibrosis",
                                        target_label='Class')

def load_page_blocks_1_3_vs_4():
    """
    Load the page-blocks-1-3_vs_4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/page-blocks-1-3_vs_4/page-blocks-1-3_vs_4.dat'
    return load_arff_template_binary(path=path,
                                        name="page-blocks-1-3_vs_4",
                                        target_label='Class')

def load_vowel0():
    """
    Load the vowel0 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/vowel0/vowel0.dat'
    return load_arff_template_binary(path=path,
                              name="vowel0",
                              target_label='Class')

def load_zoo_3():
    """
    Load the zoo-3 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/zoo-3/zoo-3.dat'
    return load_arff_template_binary(path=path,
                                        name="zoo-3",
                                        target_label='Class')

def load_haberman():
    """
    Load the haberman dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/haberman/haberman.dat'
    return load_arff_template_binary(path=path,
                                        name="haberman",
                                        target_label='Class')

def load_iris0():
    """
    Load the iris0 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/iris0/iris0.dat'
    return load_arff_template_binary(path=path,
                                        name="iris0",
                                        target_label='Class')

def load_new_thyroid1():
    """
    Load the new_thyroid1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/new_thyroid1/new-thyroid1.dat'
    return load_arff_template_binary(path=path,
                                        name="new_thyroid1",
                                        target_label='Class')

#def load_new_thyroid2():
#    """
#    Load the new_thyroid2 dataset
#
#    Returns:
#        dict: the dataset in sklearn.datasets representation
#    """
#    path = 'data/classification/new_thyroid2/new_thyroid2.dat'
#    return load_arff_template_binary(path=path,
#                                        name="new_thyroid2",
#                                        target_label='Class')

def load_page_blocks0():
    """
    Load the page_blocks0 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/page-blocks0/page-blocks0.dat'
    return load_arff_template_binary(path=path,
                                        name="new_thyroid2",
                                        target_label='Class')

def load_pima():
    """
    Load the pima dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/pima/pima.dat'
    return load_arff_template_binary(path=path,
                                        name="pima",
                                        target_label='Class')

def load_segment0():
    """
    Load the segment0 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/segment0/segment0.dat'
    return load_arff_template_binary(path=path,
                                        name="segment0",
                                        target_label='Class')

def load_wisconsin():
    """
    Load the wisconsin dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/wisconsin/wisconsin.dat'
    return load_arff_template_binary(path=path,
                                        name="wisconsin",
                                        target_label='Class')

def load_mammographic():
    """
    Load the mammographic dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/mammographic/mammographic.dat'
    return load_arff_template_binary(path=path,
                                        name="mammographic",
                                        target_label='Severity')

def load_appendicitis():
    """
    Load the appendicitis dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/appendicitis/appendicitis.dat'
    return load_arff_template_binary(path=path,
                                        name="appendicitis",
                                        target_label='Class')

def load_saheart():
    """
    Load the saheart dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/saheart/saheart.dat'
    return load_arff_template_binary(path=path,
                                        name="saheart",
                                        target_label='Chd')

def load_australian():
    """
    Load the australian dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/australian/australian.dat'
    return load_arff_template_binary(path=path,
                                        name="australian",
                                        target_label='Class')

def load_monk_2():
    """
    Load the monk-2 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/monk-2/monk-2.dat'
    return load_arff_template_binary(path=path,
                                        name='monk-2',
                                        target_label='Class')

def load_wdbc():
    """
    Load the wdbc dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/wdbc/wdbc.dat'
    return load_arff_template_binary(path=path,
                                        name="wdbc",
                                        target_label='Class')

def load_ionosphere():
    """
    Load the ionosphere dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ionosphere/ionosphere.dat'
    return load_arff_template_binary(path=path,
                                        name="ionosphere",
                                        target_label='Class')

#def load_spectfheart():
#    """
#    Load the spectfheart dataset
#
#    Returns:
#        dict: the dataset in sklearn.datasets representation
#    """
#    path = 'data/classification/spectfheart/spectfheart.dat'
#    return load_arff_template_binary(path=path,
#                                        name="spectfheart",
#                                        target_label='OVERALL_DIAGNOSIS',
#                                        revert_target=True)

def load_bupa():
    """
    Load the bupa dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/bupa/bupa.dat'
    return load_arff_template_binary(path=path,
                                        name="bupa",
                                        target_label='Selector')

def load_crx():
    """
    Load the crx dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/crx/crx.dat'
    return load_arff_template_binary(path=path,
                                        name="crx",
                                        target_label='Class')

def load_lymphography():
    """
    Load the lymphography dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/lymphography/lymphography.dat'

    dataset_raw, meta= read_arff_data(path)
    feature_types = {attr: item.type_name for attr, item in meta._attributes.items()} # pylint: disable=protected-access

    dataset_raw = pd.DataFrame(dataset_raw, columns=list(feature_types.keys()))

    target_col = dataset_raw.columns[-1]

    dataset_raw.loc[dataset_raw[target_col] == b'metastases', target_col]= 0
    dataset_raw.loc[dataset_raw[target_col] == b'malign_lymph', target_col]= 0
    dataset_raw.loc[dataset_raw[target_col] == b'normal', target_col]= 0
    dataset_raw.loc[dataset_raw[target_col] == b'fibrosis', target_col]= 1
    dataset_raw[target_col]= dataset_raw[target_col].astype(int)

    dataprep = DataPreprocessor(dataset_raw,
                            feature_types=feature_types,
                            target_label=target_col,
                            name='lymphography',
                            citation_key='keel')
    dataset = dataprep.get_dataset()

    return dataset

#######
# csv #
#######

def load_ada():
    """
    Loads the ada dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    data = read_csv_data('data/classification/ada/ada_train.data', sep=' ', header=None)
    labels = read_csv_data('data/classification/ada/ada_train.labels', header=None)

    # dummy column replaced by labels due to trailing separator characters
    data[48] = labels
    data = data.rename({48: 'target_label'}, axis='columns')

    data[data == '?'] = np.nan
    data = data.astype(float)

    return prepare_csv_data_template(dataset=data,
                        name='ADA',
                        target_label='target_label')

#def load_hiva():
#    """
#    Loads the hiva dataset
#
#    Returns:
#        dict: the dataset in sklearn.datasets representation
#    """
#    data = read_csv_data('data/classification/hiva/hiva_train.data', sep=' ', header=None)
#    labels = read_csv_data('data/classification/hiva/hiva_train.labels', header=None)
#
#    # dummy column replaced by labels due to trailing separator characters
#    data[48] = labels
#    data = data.rename({48: 'target_label'}, axis='columns')
#
#    data[data == '?'] = np.nan
#    data = data.astype(float)
#
#    return prepare_csv_data_template(dataset=data,
#                        name='HIVA',
#                        target_label='target_label')

#def load_glass():
#    """
#    Loads the glass dataset
#
#    Returns:
#        dict: the dataset in sklearn.datasets representation
#    """
#    dataset = read_csv_data('data/classification/glass/glass.data.txt')
#    dataset.columns = list(dataset.columns[:-1]) + ['target']
#    dataset.loc[dataset['target'] != 3, 'target'] = 0
#
#    return prepare_csv_data_template(dataset=dataset,
#                        name='glass',
#                        target_label='target')

def load_hypothyroid():
    """
    Loads the hypothyroid dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """

    dataset = read_csv_data('data/classification/hypothyroid/hypothyroid.data.txt')

    dataset.columns= [str(col) for col in dataset.columns]

    dataset[dataset == '?'] = np.nan
    dataset[dataset == 'f'] = 0
    dataset[dataset == 't'] = 1
    dataset[dataset == 'F'] = 0
    dataset[dataset == 'T'] = 1
    dataset[dataset == 'n'] = 0
    dataset[dataset == 'y'] = 1
    dataset[dataset == 'M'] = 0
    dataset[dataset == 'F'] = 1
    dataset[dataset.columns[1:]] = dataset[dataset.columns[1:]].astype(float)
    dataset.columns = ['target'] + list(dataset.columns[1:])

    return prepare_csv_data_template(dataset=dataset,
                        name='hypothyroid',
                        target_label='target')

def load_sylva():
    """
    Loads the sylva dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    database_raw= read_csv_data('data/classification/sylva/sylva_train.data', sep= ' ')

    # removing last column due to trailing whitespaces
    del database_raw[database_raw.columns[-1]]

    target= read_csv_data('data/classification/sylva/sylva_train.labels')
    database_raw['target']= target

    return prepare_csv_data_template(dataset=database_raw,
                        name='hypothyroid',
                        target_label='target')

def load_spectf():
    """
    Loads the spectf dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    db0= read_csv_data('data/classification/spect_f/SPECTF.train.txt')
    db1= read_csv_data('data/classification/spect_f/SPECTF.test.txt')
    dataset= pd.concat([db0, db1])
    dataset.columns= ['target'] + list(dataset.columns[1:])

    return prepare_csv_data_template(dataset=dataset,
                        name='SPECTF',
                        target_label='target')

def load_hepatitis():
    """
    Loads the hepatitis dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    dataset= read_csv_data('data/classification/hepatitis/hepatitis.data.txt')
    dataset.columns= ['target'] + list(dataset.columns[1:])

    dataset[dataset == '?'] = np.nan
    dataset = dataset.astype(float)

    return prepare_csv_data_template(dataset=dataset,
                        name='hepatitis',
                        target_label='target')

#def load_vehicle():
#    """
#    Loads the vehicle dataset
#
#    Returns:
#        dict: the dataset in sklearn.datasets representation
#    """
#    db0= read_csv_data('data/classification/vehicle/xaa.dat.txt', sep= ' ', usecols= range(19))
#    db1= read_csv_data('data/classification/vehicle/xab.dat.txt', sep= ' ', usecols= range(19))
#    db2= read_csv_data('data/classification/vehicle/xac.dat.txt', sep= ' ', usecols= range(19))
#    db3= read_csv_data('data/classification/vehicle/xad.dat.txt', sep= ' ', usecols= range(19))
#    db4= read_csv_data('data/classification/vehicle/xae.dat.txt', sep= ' ', usecols= range(19))
#    db5= read_csv_data('data/classification/vehicle/xaf.dat.txt', sep= ' ', usecols= range(19))
#    db6= read_csv_data('data/classification/vehicle/xag.dat.txt', sep= ' ', usecols= range(19))
#    db7= read_csv_data('data/classification/vehicle/xah.dat.txt', sep= ' ', usecols= range(19))
#    db8= read_csv_data('data/classification/vehicle/xai.dat.txt', sep= ' ', usecols= range(19))
#
#    dataset= pd.concat([db0, db1, db2, db3, db4, db5, db6, db7, db8])
#
#    dataset.columns= list(dataset.columns[:-1]) + ['target']
#    dataset.loc[dataset['target'] != 'van', 'target']= 'other'
#
#    return prepare_csv_data_template(dataset=dataset,
#                        name='vehicle',
#                        target_label='target')

def load_german():
    """
    Loads the german dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """

    dataset = read_csv_data('data/classification/german/german.data-numeric.txt', sep= '\t')
    dataset.columns= list(dataset.columns[:-1]) + ['target']

    return prepare_csv_data_template(dataset=dataset,
                        name='german',
                        target_label='target')

def load_satimage():
    """
    Loads the satimage dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """

    db0= read_csv_data('data/classification/satimage/sat.trn.txt', sep= ' ')
    db1= read_csv_data('data/classification/satimage/sat.tst.txt', sep= ' ')
    dataset= pd.concat([db0, db1])
    dataset.columns= list(dataset.columns[:-1]) + ['target']
    dataset.loc[dataset['target'] != 4, 'target']= 0

    return prepare_csv_data_template(dataset=dataset,
                        name='SATIMAGE',
                        target_label='target')
