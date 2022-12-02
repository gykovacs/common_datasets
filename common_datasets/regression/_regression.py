"""
This module contains the regression loaders.
"""

import json
import io
import pkgutil

import pandas as pd

from .._io import (read_csv_data, read_xls_data,
                        load_arff_template_regression,
                        prepare_csv_data_template,
                        prepare_xls_data_template)

summary = json.loads(pkgutil.get_data('common_datasets', 'data/summary_regression.json').decode('utf-8'))

summary_pdf = pd.DataFrame.from_dict(summary)

__all__= ['load_airfoil',
          'load_cpu_performance',
          'load_forestfires',
          'load_real_estate_valuation',
          'load_residential_building',
          'load_slump_test',
          'load_stock_portfolio_performance',
          'load_winequality_red',
          'load_winequality_white',
          'load_yacht_hydrodynamics',
          'load_ccpp',
          'load_communities',
          'load_diabetes',
          'load_laser',
          'load_autoMPG6',
          'load_wizmir',
          'load_wankara',
          'load_mortgage',
          'load_baseball',
          'load_treasury',
          'load_compactiv',
          'load_puma32h',
          'get_data_loaders',
          'get_filtered_data_loaders',
          'summary_pdf']

########
# arff #
########

def load_diabetes():
    """
    Load the diabetes dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/diabetes/diabetes.dat',
                                            name='diabetes',
                                            target_label='C_peptide')

def load_laser():
    """
    Load the laser dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/laser/laser.dat',
                                            name='laser',
                                            target_label='Output')

def load_autoMPG6(): # pylint: disable=invalid-name
    """
    Load the autoMPG6 dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/autoMPG6/autoMPG6.dat',
                                            name='autoMPG6',
                                            target_label='Mpg')

def load_wizmir():
    """
    Load the wizmir dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/wizmir/wizmir.dat',
                                            name='wizmir',
                                            target_label='Mean_temperature')

def load_wankara():
    """
    Load the wankara dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/wankara/wankara.dat',
                                            name='wankara',
                                            target_label='Mean_temperature')

def load_mortgage():
    """
    Load the mortgage dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/mortgage/mortgage.dat',
                                            name='mortgage',
                                            target_label='30Y-CMortgageRate')

def load_baseball():
    """
    Load the baseball dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/baseball/baseball.dat',
                                            name='baseball',
                                            target_label='Salary')

def load_treasury():
    """
    Load the treasury dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/treasury/treasury.dat',
                                            name='treasury',
                                            target_label='1MonthCDRate')

def load_compactiv():
    """
    Load the compactiv dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/compactiv/compactiv.dat',
                                            name='compactiv',
                                            target_label='usr')

def load_puma32h():
    """
    Load the puma32h dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    return load_arff_template_regression(path='data/regression/puma32h/puma32h.dat',
                                            name='puma32h',
                                            target_label='thetadd6')

#######
# csv #
#######

def load_airfoil():
    """
    Load the airfoil dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    dataset = read_csv_data('data/regression/airfoil/airfoil_self_noise.dat.txt',
                        sep='\t')
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_csv_data_template(dataset=dataset,
                        name='airfoil',
                        target_label='target',
                        problem_type='regression')

def load_cpu_performance():
    """
    Load the cpu_performance dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    dataset = read_csv_data('data/regression/cpu_performance/machine.data.txt',
                        sep=',')
    del dataset[dataset.columns[-1]]
    del dataset[dataset.columns[1]]
    columns  = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_csv_data_template(dataset=dataset,
                        name='cpu_performance',
                        target_label='target',
                        problem_type='regression')

def load_forestfires():
    """
    Load the forestfires dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    dataset = read_csv_data('data/regression/forestfires/forestfires.csv',
                        sep=',',
                        header=0)
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_csv_data_template(dataset=dataset,
                        name='forestfires',
                        target_label='target',
                        problem_type='regression')

def load_slump_test():
    """
    Load the slump_test dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    dataset= read_csv_data('data/regression/slump_test/slump_test.data.txt',
                        sep=',',
                        header=0)
    del dataset[dataset.columns[0]]
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_csv_data_template(dataset=dataset,
                        name='slump_test',
                        target_label='target',
                        problem_type='regression')

def load_winequality_red():
    """
    Load the winequality_red dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    dataset = read_csv_data('data/regression/winequality_red/winequality-red.csv',
                        sep=';',
                        header=0)
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_csv_data_template(dataset=dataset,
                        name='winequality_red',
                        target_label='target',
                        problem_type='regression')

def load_winequality_white():
    """
    Load the winequality_white dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    dataset = read_csv_data('data/regression/winequality_white/winequality-white.csv',
                        sep=';',
                        header=0)
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_csv_data_template(dataset=dataset,
                        name='winequality_white',
                        target_label='target',
                        problem_type='regression')

def load_yacht_hydrodynamics():
    """
    Load the yacht_hydrodynamics dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    dataset = read_csv_data('data/regression/yacht_hydrodynamics/yacht_hydrodynamics.data.txt',
                        sep=None,
                        header=0,
                        delim_whitespace=True)
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_csv_data_template(dataset=dataset,
                        name='yacht_hydrodynamics',
                        target_label='target',
                        problem_type='regression')

def load_communities():
    """
    Load the communities dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    dataset = read_csv_data('data/regression/communities/communities.data',
                        sep=',')
    dataset = dataset.drop(3, axis='columns')
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_csv_data_template(dataset=dataset,
                        name='communities',
                        target_label='target',
                        problem_type='regression')

########
# xlsx #
########

def load_real_estate_valuation():
    """
    Load the real_estate_valuation dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    filename = 'data/regression/real_estate_valuation/Real estate valuation data set.xlsx'
    dataset = read_xls_data(filename)

    del dataset[dataset.columns[0]]
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    for col in dataset.columns:
        dataset[col] = dataset[col].astype(float)

    return prepare_xls_data_template(dataset=dataset,
                        name="real_estate_valuation",
                        target_label='target')

def load_residential_building():
    """
    Load the residential_building dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    # target: V9

    filename = 'data/regression/residential_building/Residential-Building-Data-Set.xlsx'

    dataset = read_xls_data(filename)
    dataset = dataset.drop(0, axis='index')
    dataset.reset_index(drop=True, inplace=True)
    del dataset[dataset.columns[-1]]
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    for col in dataset.columns:
        dataset[col] = dataset[col].astype(float)

    return prepare_xls_data_template(dataset=dataset,
                        name="residential_building",
                        target_label='target')

def load_stock_portfolio_performance():
    """
    Load the stock_portfolio_performance dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """
    # target: normalized annual return

    filename = 'data/regression/stock_portfolio_performance/stock '\
                'portfolio performance data set.xlsx'
    dataset = read_xls_data(filename,
                        sheet_name='all period')

    dataset.columns = dataset.iloc[0].values
    dataset = dataset.drop(dataset.index[0], axis='index')
    dataset.reset_index(drop=True, inplace=True)
    del dataset[dataset.columns[0]]
    columns = list(dataset.columns[0:5]) + [dataset.columns[11]]
    dataset = dataset[columns]
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    for col in dataset.columns:
        dataset[col] = dataset[col].astype(float)

    return prepare_xls_data_template(dataset=dataset,
                        name="stock_portfolio_performance",
                        target_label='target')

def load_ccpp():
    """
    Load the ccpp dataset

    Returns:
        dict: the dataset in sklearn.datasets format
    """

    dataset = read_xls_data('data/regression/ccpp/Folds5x2_pp.xlsx',
                        sheet_name='Sheet1')
    columns = list(dataset.columns)
    columns[-1] = 'target'
    dataset.columns = columns

    return prepare_xls_data_template(dataset=dataset,
                        name="ccpp",
                        target_label='target')

#########
# other #
#########

def get_filtered_data_loaders(*,
                              n_col_bounds=(1, 5000),
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
        sorting (str): the sorting attribute ('n', 'n_col')
        n_from_phenotypes (int): the maximum number of datasets from a
                                    phenotype

    Returns:
        list: the list of data loaders
    """
    descriptors = summary_pdf
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
