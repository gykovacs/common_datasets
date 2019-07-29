#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 14:41:22 2019

@author: gykovacs
"""

import copy
import sys
import pkgutil
import io

# for the representation of the data
import numpy as np
import pandas as pd

import logging

# for the encoding of the data
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from scipy.io import arff

from statistics import mode

__all__=['encode_features',
         'construct_return_set',
         'read_csv_data',
         'read_arff_data',
         'read_xls_data']

_logger= logging.getLogger('mldb')
_logger.setLevel(logging.INFO)
_logger_ch= logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)

citations= {
'krnn': """@article{krnn,
author={X. J. Zhang and Z. Tari and M. Cheriet},
title={{KRNN}: k {Rare-class Nearest Neighbor} classification},
journal={Pattern Recognition},
year={2017},
volume={62},
number={2},
pages={33--44}
}""",
'keel':"""@article{keel,
author={Alcala-Fdez, J. and Fernandez, A. and Luengo, J. and Derrac, J. and Garcia, S. and Sanchez, L. and Herrera, F.},
title={KEEL Data-Mining Software Tool: Data Set Repository, Integration of Algorithms and Experimental Analysis Framework},
journal={Journal of Multiple-Valued Logic and Soft Computing},
volume={17},
number={2-3},
year={2011},
pages={255-287}
}
""",
'uci':"""
@misc{uci,
author = "Dua, Dheeru and Karra Taniskidou, Efi",
year = "2017",
title = "{UCI} Machine Learning Repository",
url = "http://archive.ics.uci.edu/ml",
institution = "University of California, Irvine, School of Information and Computer Sciences" }
"""
}

def encode_column_onehot(column):
    """
    Applies one-hot encoding to a column.
    
    Args:
        column (pd.Series): the column to be encoded
    
    Returns:
        np.ndarray: the encoded column
    """
    ohencoder= OneHotEncoder(sparse= False, categories='auto').fit(column.values.reshape(-1, 1))
    ohencoded= ohencoder.transform(column.values.reshape(-1, 1))
    
    return ohencoded.astype(float)

def encode_column_ordinal(column):
    """
    Applies ordinal encoding to a column.
    
    Args:
        column (pd.Series): the column to be encoded
        
    Returns:
        np.array: the encoded column
    """
    oencoder= OrdinalEncoder(categories='auto').fit(column.values.reshape(-1, 1))
    return oencoder.transform(column.values.reshape(-1, 1)).astype(float)[:,0]

def encode_column_median(column, missing_values):
    """
    Replaces missing values in the column by medians.
    
    Args:
        column (pd.Series): the column to be encoded
    Returns:
        np.array: the encoded, missing-value-free column
    """
    column= copy.deepcopy(column)
    if np.sum(column.isin(missing_values)) > 0:
        column[column.isin(missing_values)]= np.median(column[~column.isin(missing_values)].astype(float))
    column= column.astype(float)
    return column.values

def encode_features(data, target= 'target', encoding_threshold= 5, missing_values= ['?', None, 'None'], problem_type= 'imbalanced', verbose= True):
    """
    Automated feature encoding
    
    Args:
        data (pd.DataFrame): pd.DataFrame containing the dataset
        target (str): name of the target label
        encoding_threshold (int): the number of distinct values at or below one-hot encoding is applied
        missing_values (list): the list of missing value placeholders
        problem_type (str): 'imbalanced'/'multiclass'/'regression'/'clustering'
        verbose (boolean): True for verbose output
    Returns:
        pd.DataFrame: the encoded dataset
    """
    if verbose:
        logging.getLogger('mldb').setLevel(logging.INFO)
    else:
        logging.getLogger('mldb').setLevel(logging.WARNING)
    
    columns= []
    column_names= []
    
    for c in data.columns:
        logging.info('Encoding column %s' % c)
        if not c == target:
            # if the column is not the target variable
            n_values= len(np.unique(data[c]))
            logging.info(' number of values: %d => ' % n_values)
            
            convertible_to_float= False
            try:
                data[c][~data[c].isin(missing_values)].astype(float)
                convertible_to_float= True
            except:
                pass
            
            if n_values == 1:
                # there is no need for encoding
                logging.info('no encoding')
                continue
            if (n_values == 2 or data[c].dtype == object) and convertible_to_float is False:
                # applying label encoding
                logging.info('ordinal encoding')
                columns.append(encode_column_ordinal(data[c]))
                column_names.append(c)
            elif n_values < encoding_threshold:
                # applying one-hot encoding
                logging.info('one-hot encoding')
                ohencoded= encode_column_onehot(data[c])
                for i in range(ohencoded.shape[1]):
                    columns.append(ohencoded[:,i])
                    column_names.append(str(c) + '_onehot_' + str(i))
            else:
                # applying median encoding
                logging.info('no encoding, missing values replaced by medians')
                columns.append(encode_column_median(data[c], missing_values))
                column_names.append(c)
                
        if c == target:
            if problem_type == 'imbalanced':
                # in the target column the least frequent value is set to 1, the
                # rest is set to 0
                logging.info(' target variable => least frequent value is set to 1')
                column= copy.deepcopy(data[c])
                val_counts= data[target].value_counts()
                if val_counts.values[0] < val_counts.values[1]:
                    mask= (column == val_counts.index[0])
                    column[mask]= 1
                    column[~(mask)]= 0
                else:
                    mask= (column == val_counts.index[0])
                    column[mask]= 0
                    column[~(mask)]= 1
    
                columns.append(column.astype(int).values)
                column_names.append(target)
            elif problem_type == 'multiclass':
                logging.info('multiclass encoding of target variable')
                columns.append(LabelEncoder().fit_transform(data[c]).astype(int))
                column_names.append(target)
            elif problem_type == 'regression':
                columns.append(data[c].values)
                column_names.append(target)
            elif problem_type == 'clustering':
                pass
            else:
                raise ValueError('Problem type "%s" not implemented' % problem_type)
    
    dataset= pd.DataFrame(np.vstack(columns).T, columns= column_names)
    if problem_type == 'imbalanced' or problem_type == 'multiclass':
        dataset[target]= dataset[target].astype(int)
    return dataset

def construct_return_set(database, 
                         descriptor, 
                         return_X_y, 
                         encode, 
                         encoding_threshold= 5, 
                         problem_type= 'imbalanced',
                         target_name= 'target',
                         citation= None, 
                         name= None,
                         verbose= True):
    
    """
    Constructs the return set of the database
    
    Args:
        database (pd.DataFrame): database to be encoded
        descriptor (str): description of the database
        return_X_y (boolean): if True, only the independent and dependent features are returned
        encode (boolean): if True, encoding is applied
        encoding_threshold (int): the threshold of one-hot/ordinal encoding
        problem_type (str): 'imbalanced'/'multiclass'/'regression'/'clustering'
        target_name (str): name of the target variable
        citation (str): citation to include
        name (str): name to include
        verbose (boolean): True for verbose output
    
    Returns:
        dict: the prepared dataset
    """

    if return_X_y == True and encode == False:
        return database.drop(target_name, axis= 'columns').values, database[target_name].values
    
    if encode == True:
        database= encode_features(database, encoding_threshold= encoding_threshold, verbose= verbose, problem_type=problem_type)
        for c in database.columns:
            if not c is target_name:
                database[c]= database[c].astype(float)
        if return_X_y == True:
            return database.drop(target_name, axis= 'columns').values, database[target_name].values

    descriptors= {}
    descriptors['DESCR']= descriptor
    features= database.drop(target_name, axis= 'columns')
    descriptors['data']= features.values
    descriptors['feature_names']= list(features.columns)
    descriptors['target']= database[target_name].values
    descriptors['citation']= citation
    descriptors['name']= name
    
    return descriptors

def read_csv_data(filename, sep= ',', usecols= None, header= None, delim_whitespace=False):
    if delim_whitespace:
        return pd.read_csv(io.BytesIO(pkgutil.get_data('mldb', filename)), header= header, usecols= usecols, delim_whitespace=delim_whitespace)
    else:
        return pd.read_csv(io.BytesIO(pkgutil.get_data('mldb', filename)), header= header, usecols= usecols, sep=sep)

def read_xls_data(filename, header= None, sheet_name= None):
    if sheet_name is None:
        return pd.read_excel(io.BytesIO(pkgutil.get_data('mldb', filename)))
    else:
        return pd.read_excel(io.BytesIO(pkgutil.get_data('mldb', filename)), sheet_name= sheet_name)

def read_arff_data(filename, sep= ',', usecols= None):
    if sys.version_info >= (3,0):
        return arff.loadarff(io.StringIO(pkgutil.get_data('mldb', filename).decode('unicode_escape')))
    else:
        from cStringIO import StringIO
        return arff.loadarff(StringIO(unicode(str(pkgutil.get_data('mldb', filename)).decode('string_escape'), "utf-8")))

