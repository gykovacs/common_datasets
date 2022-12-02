"""
This module contains the multiclass data loaders
"""

import json
import io
import pkgutil

import pandas as pd

from .._io import (read_csv_data,
                    load_arff_template_multiclass,
                    prepare_csv_data_template)

summary = json.loads(pkgutil.get_data('common_datasets', 'data/summary_multiclass_classification.json').decode('utf-8'))

summary_pdf = pd.DataFrame.from_dict(summary)

__all__= ['load_glass',
          'load_satimage',
          'load_ecoli',
          'load_abalone',
          'load_yeast',
          'load_automobile',
          'load_balance',
          'load_car',
          'load_cleveland',
          'load_contraceptive',
          'load_dermatology',
          'load_fars',
          'load_flare',
          'load_hayes_roth',
          'load_kr_vs_k',
          'load_led7digit',
          'load_movement_libras',
          'load_newthyroid',
          'load_nursery',
          'load_page_blocks',
          'load_post_operative',
          'load_segment',
          'load_splice',
          'load_tae',
          'load_vowel',
          'load_zoo',
          'get_filtered_data_loaders',
          'get_data_loaders',
          'summary_pdf']

########
# arff #
########

def load_automobile():
    """
    Loads the automobile dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/automobile/automobile.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Symboling',
                                            name='automobile')

def load_balance():
    """
    Loads the balance dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/balance/balance.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Balance_scale',
                                            name='balance')

def load_car():
    """
    Loads the car dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/car/car.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Acceptability',
                                            name='car')

def load_cleveland():
    """
    Loads the cleveland dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/cleveland/cleveland.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Num',
                                            name='cleveland')

def load_contraceptive():
    """
    Loads the contraceptive dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/contraceptive/contraceptive.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Contraceptive_method',
                                            name='contraceptive')

def load_dermatology():
    """
    Loads the dermatology dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/dermatology/dermatology.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='dermatology')

def load_fars():
    """
    Loads the fars dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/fars/fars.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='INJURY_SEVERITY',
                                            name='fars')

def load_flare():
    """
    Loads the flare dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/flare/flare.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='flare')

def load_hayes_roth():
    """
    Loads the hayes_roth dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/hayes-roth/hayes-roth.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='hayes_roth')

def load_kr_vs_k():
    """
    Loads the kr_vs_k dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/kr-vs-k/kr-vs-k.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Game',
                                            name='kr_vs_k')

def load_led7digit():
    """
    Loads the led7digit dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/led7digit/led7digit.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Number',
                                            name='led7digit')

def load_movement_libras():
    """
    Loads the movement_libras dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/movement_libras/movement_libras.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='movement_libras')

def load_newthyroid():
    """
    Loads the newthyroid dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/newthyroid/newthyroid.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='newthyroid')

def load_nursery():
    """
    Loads the nursery dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/nursery/nursery.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='nursery')

def load_page_blocks():
    """
    Loads the page_blocks dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/page-blocks/page-blocks.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='page_blocks')

def load_post_operative():
    """
    Loads the post_operative dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/post-operative/post-operative.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Decision',
                                            name='post_operative')

def load_segment():
    """
    Loads the segment dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/segment/segment.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='segment')

def load_splice():
    """
    Loads the splice dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/splice/splice.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='splice')

def load_tae():
    """
    Loads the tae dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/tae/tae.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='tae')

def load_vowel():
    """
    Loads the vowel dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/vowel/vowel.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Class',
                                            name='vowel')

def load_zoo():
    """
    Loads the zoo dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    path = 'data/classification/zoo/zoo.dat'
    return load_arff_template_multiclass(path=path,
                                            target_label='Type',
                                            name='zoo')

#######
# csv #
#######

def load_glass():
    """
    Load the glass dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    dataset= read_csv_data('data/classification/glass/glass.data.txt')
    dataset.columns= list(dataset.columns[:-1]) + ['target']
    del dataset[dataset.columns[0]]

    return prepare_csv_data_template(dataset=dataset,
                        name='glass',
                        target_label='target',
                        problem_type='multiclass')

def load_satimage():
    """
    Load the SATIMAGE dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    db0= read_csv_data('data/classification/satimage/sat.trn.txt',
                        sep= ' ')
    db1= read_csv_data('data/classification/satimage/sat.tst.txt',
                        sep= ' ')

    dataset= pd.concat([db0, db1])
    dataset.columns= list(dataset.columns[:-1]) + ['target']

    return prepare_csv_data_template(dataset=dataset,
                        name='SATIMAGE',
                        target_label='target',
                        problem_type='multiclass')

def load_ecoli():
    """
    Load the ecoli dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    dataset= read_csv_data('data/classification/ecoli/ecoli.data.txt',
                        delim_whitespace=True)
    dataset.columns= list(dataset.columns[:-1]) + ['target']
    del dataset[dataset.columns[0]]

    return prepare_csv_data_template(dataset=dataset,
                        name='ecoli',
                        target_label='target',
                        problem_type='multiclass')

def load_abalone():
    """
    Load the abalone dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    dataset= read_csv_data('data/classification/abalone/abalone.data.txt')
    dataset.columns= list(dataset.columns[:-1]) + ['target']
    del dataset[dataset.columns[0]]

    return prepare_csv_data_template(dataset=dataset,
                        name='abalone',
                        target_label='target',
                        problem_type='multiclass')

def load_yeast():
    """
    Load the yeast dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    dataset= read_csv_data('data/classification/yeast/yeast.data.txt',
                        delim_whitespace=True)
    dataset.columns= list(dataset.columns[:-1]) + ['target']
    del dataset[dataset.columns[0]]

    return prepare_csv_data_template(dataset=dataset,
                        name='yeast',
                        target_label='target',
                        problem_type='multiclass')

#############################

def get_filtered_data_loaders(n_col_bounds=(1, 5000),
                              n_col_orig_bounds=(1, 5000),
                              n_bounds=(1, 10000),
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
        n_smallest (int): the number of smallest in the sense of "sorting"
        sorting (str): the sorting attribute ('n', 'n_col', 'n_minority',
                            'imbalance_ratio')
        n_from_phenotypes (bool): the maximum number of datasets from a
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
                                & (descriptors['n_col_orig'] < n_col_orig_bounds[1])]

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

    if subset == 'study':
        n_col_bounds = (n_col_bounds[0], 100)
        n_bounds = (n_bounds[0], 4000)
    elif subset == 'small':
        n_col_bounds = (n_col_bounds[0], 100)
        n_bounds = (n_bounds[0], 1000)
    elif subset == 'tiny':
        n_bounds = (n_bounds[0], 120)

    return get_filtered_data_loaders(n_col_bounds=n_col_bounds,
                                    n_col_orig_bounds=n_col_orig_bounds,
                                    n_bounds=n_bounds,
                                    n_smallest=n_smallest,
                                    sorting=sorting,
                                    n_from_phenotypes=n_from_phenotypes)
