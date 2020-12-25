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
         'read_xls_data',
         'references']

_logger= logging.getLogger('mldb')
_logger.setLevel(logging.INFO)
_logger_ch= logging.StreamHandler()
_logger_ch.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s:%(message)s"))
_logger.addHandler(_logger_ch)

references= {
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

import logging

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

class SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, missing_values, add_indicator=False):
        self.strategy=strategy
        self.missing_values=missing_values
        self.add_indicator=add_indicator
    
    def fit(self, X, y=None):
        X= X.values
        self.indicator_needed= np.sum([np.sum((X == m)) for m in self.missing_values]) > 0
        return self

    def transform(self, X, y=None):
        X= X.values
        missing_mask= (X == self.missing_values[0])
        for m in self.missing_values[1:]:
            missing_mask= np.logical_or(missing_mask, (X == m))
        if self.strategy == 'median':
            fill_value= np.median(X[np.logical_not(missing_mask)])
        elif self.strategy == 'most_frequent':
            values, counts= np.unique(X[np.logical_not(missing_mask)], return_counts=True)
            fill_value= values[np.argmax(counts)]
        
        X[missing_mask]= fill_value
        if self.indicator_needed:
            indicator= missing_mask.astype(float)
            X= np.c_[X, indicator]
        
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class ClassLabelEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        values, counts= np.unique(X, return_counts=True)
        pairs= list(zip(values, counts))
        pairs= sorted(pairs, key=lambda x: -x[1])

        masks= {}
        for i in range(len(pairs)):
            masks[i]= (X == pairs[i][0])

        for i in masks:
            X[masks[i]]= i
        
        return X.astype(int)

class EncoderAndImputer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                    target_attribute= None,
                    problem_type='classification',
                    onehot_limit= 10,
                    numeric_strategy='median',
                    onehot_strategy='most_frequent',
                    ordinal_strategy='most_frequent',
                    missing_values_numeric=[None, np.nan],
                    missing_values_string=['?', None, 'None']):
        self.target_attribute=target_attribute
        self.problem_type=problem_type
        self.onehot_limit= onehot_limit
        self.numeric_strategy=numeric_strategy
        self.onehot_strategy=onehot_strategy
        self.ordinal_strategy=ordinal_strategy
        self.missing_values_numeric= missing_values_numeric
        self.missing_values_string=missing_values_string
    
    def fit(self, X, column_specifications=None):
        self.encoders= {}
        self.imputers= {}

        for c in X.columns:
            n_uniques= len(pd.unique(X[c]))
            if n_uniques > 1:
                if not self.target_attribute or (self.target_attribute and not c == self.target_attribute):
                    if np.issubdtype(X[c].dtype, np.number) or (column_specifications and column_specifications[c] == 'numeric'):
                        logging.info("%s: %s" % (c, 'numeric'))
                        self.imputers[c]= SimpleImputer(strategy="median", add_indicator=True, missing_values=self.missing_values_numeric)
                        self.encoders[c]= StandardScaler()
                    elif np.issubdtype(X[c].dtype, np.dtype(object).type):
                        if n_uniques > 2 and n_uniques <= self.onehot_limit or (column_specifications and column_specifications[c] == 'onehot'):
                            logging.info("%s: %s" % (c, 'onehot'))
                            self.imputers[c]= SimpleImputer(strategy='most_frequent', add_indicator=True, missing_values=self.missing_values_string)
                            self.encoders[c]= OneHotEncoder()
                        else:
                            logging.info("%s: %s" % (c, 'ordinal'))
                            self.imputers[c]= SimpleImputer(strategy='most_frequent', add_indicator=True, missing_values=self.missing_values_string)
                            self.encoders[c]= OrdinalEncoder()
                    else:
                        logging.info("%s: %s" % (c, 'no encoding'))
                        self.imputers[c]= None
                        self.encoders[c]= None
                else:
                    if self.target_attribute:
                        if self.problem_type == 'regression':
                            logging.info("%s: %s" % (c, 'regression target encoding'))
                            self.target_imputer= SimpleImputer(strategy='median', add_indicator=False, missing_values= [None, np.nan])
                            self.target_encoder= None
                        elif self.problem_type == 'classification':
                            logging.info("%s: %s" % (c, 'classification target encoding'))
                            self.target_imputer= SimpleImputer(strategy='most_frequent', add_indicator=False, missing_values= [None])
                            self.target_encoder= ClassLabelEncoder()
                    else:
                        logging.info("%s: %s" % (c, 'no values'))

        for c in self.encoders:
            logging.info("fitting encoder for attribute %s" % c)
            if self.encoders[c]:
                if np.issubdtype(X[c].dtype, np.dtype(object).type):
                    imputed= self.imputers[c].fit_transform(X[[c]].astype(str))
                    self.encoders[c].fit(imputed[:,[0]].astype(str))
                else:
                    imputed= self.imputers[c].fit_transform(X[[c]])
                    self.encoders[c].fit(imputed[:,[0]])
        
        if self.target_attribute:
            if self.target_imputer:
                if np.issubdtype(X[self.target_attribute].dtype, np.dtype(object).type):
                    imputed= self.target_imputer.fit_transform(X[[self.target_attribute]].astype(str))
                    if self.target_encoder:
                        self.target_encoder.fit(imputed)
                else:
                    imputed= self.target_imputer.fit_transform(X[[self.target_attribute]])
                    if self.target_encoder:
                        self.target_encoder.fit(imputed)

        return self
    
    def transform(self, X):
        X_enc, cols= [], []
        for c in self.encoders:
            if self.encoders[c]:
                if np.issubdtype(X[c].dtype, np.dtype(object).type):
                    imputed= self.imputers[c].transform(X[[c]].astype(str))
                    if isinstance(self.encoders[c], OneHotEncoder):
                        encoded= self.encoders[c].transform(imputed[:,[0]]).todense()
                    else:
                        encoded= self.encoders[c].transform(imputed[:,[0]])
                else:
                    imputed= self.imputers[c].transform(X[[c]])
                    if isinstance(self.encoders[c], OneHotEncoder):
                        encoded= self.encoders[c].transform(imputed[:,[0]]).todense()
                    else:
                        encoded= self.encoders[c].transform(imputed[:,[0]])
                X_enc.append(encoded)
                if encoded.shape[1] > 1:
                    cols.extend([str(c) + "_onehot_" + str(i) for i in range(encoded.shape[1])])
                else:
                    cols.append(c)
                if imputed.shape[1] == 2:
                    X_enc.append(imputed[:,[1]])
                    cols.append(str(c) + '_missing_value_indicator')
            else:
                X_enc.append(X[[c]].values)
                cols.append(c)
        
        data= np.hstack(X_enc)

        if hasattr(self, 'scaler'):
            data= self.scaler.transform(data)
        else:
            self.scaler= StandardScaler()
            data= self.scaler.fit_transform(data)
        
        data= pd.DataFrame(data, columns=cols)
        
        if self.target_attribute:
            if np.issubdtype(X[self.target_attribute].dtype, np.dtype(object).type):
                imputed= self.target_imputer.transform(X[[self.target_attribute]].astype(str))
            else:
                imputed= self.target_imputer.transform(X[[self.target_attribute]])
            if self.target_encoder:
                encoded= self.target_encoder.transform(imputed)
            else:
                encoded= imputed

            data[self.target_attribute]= encoded
        
        return data
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def construct_return_set(database, 
                         descriptor, 
                         return_X_y, 
                         encode, 
                         problem_type= 'classification',
                         target_name= 'target',
                         citation= None, 
                         name= None,
                         verbose= True,
                         onehot_threshold= 10):
    
    """
    Constructs the return set of the database
    
    Args:
        database (pd.DataFrame): database to be encoded
        descriptor (str): description of the database
        return_X_y (boolean): if True, only the independent and dependent features are returned
        encode (boolean): if True, encoding is applied
        encoding_threshold (int): the threshold of one-hot/ordinal encoding
        problem_type (str): 'classification'/'regression'/'clustering'
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
        database_encoded= EncoderAndImputer(onehot_limit=onehot_threshold, problem_type=problem_type, target_attribute='target').fit_transform(database)
        for c in database_encoded.columns:
            if not c is target_name:
                database_encoded[c]= database_encoded[c].astype(float)
        
        if return_X_y == True:
            return database_encoded.drop(target_name, axis= 'columns').values, database_encoded[target_name].values

    descriptors= {}
    descriptors['DESCR']= descriptor
    if encode == True:
        features= database_encoded.drop(target_name, axis= 'columns')
        descriptors['data']= features.values
        descriptors['feature_names']= list(features.columns)
        descriptors['target']= database_encoded[target_name].values
        features= database.drop(target_name, axis='columns')
        descriptors['data_raw']= features.values
        descriptors['feature_names_raw']= list(features.columns)
        descriptors['target_raw']= database[target_name].values
    else:
        features= database.drop(target_name, axis='columns')
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
        return pd.read_excel(io.BytesIO(pkgutil.get_data('mldb', filename)), engine='openpyxl')
    else:
        return pd.read_excel(io.BytesIO(pkgutil.get_data('mldb', filename)), sheet_name= sheet_name, engine='openpyxl')

def read_arff_data(filename, sep= ',', usecols= None):
    if sys.version_info >= (3,0):
        return arff.loadarff(io.StringIO(pkgutil.get_data('mldb', filename).decode('unicode_escape')))
    else:
        from cStringIO import StringIO
        return arff.loadarff(StringIO(unicode(str(pkgutil.get_data('mldb', filename)).decode('string_escape'), "utf-8")))

