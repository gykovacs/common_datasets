"""
This module contains all the IO functionalities
"""

import pkgutil
import io
import logging

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from scipy.io import arff

__all__=['read_csv_data',
         'read_arff_data',
         'read_xls_data',
         'references',
         'coalesce',
         'IdentityTransformer',
         'LabelEncoder',
         'determine_types',
         'prepare_csv_data_template',
         'prepare_xls_data_template',
         'load_arff_template',
         'load_arff_template_binary',
         'load_arff_template_multiclass',
         'load_arff_template_regression',
         'numeric_preprocessing',
         'category_preprocessing',
         'ordinal_preprocessing',
         'class_label_preprocessing',
         'multiclass_label_preprocessing',
         'DataPreprocessor']

_logger= logging.getLogger('common_datasets')
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
""",
'boom_bikes':"""
@article{boombikes,
	year={2013},
	issn={2192-6352},
	journal={Progress in Artificial Intelligence},
	doi={10.1007/s13748-013-0040-3},
	title={Event labeling combining ensemble detectors and background knowledge},
	url={http://dx.doi.org/10.1007/s13748-013-0040-3},
	publisher={Springer Berlin Heidelberg},
	keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
	author={Fanaee-T, Hadi and Gama, Joao},
	pages={1-15}
}
""",
'mlwithr':"""
@book{mlwithr,
author = {Lantz, Brett},
title = {Machine Learning with R},
year = {2013},
isbn = {1782162143},
publisher = {Packt Publishing}
}
"""
}

# Revisit later
#
#def adjusted_median(array):
#    if (array.shape[0] % 2) == 1:
#        return np.median(array)
#
#    values, counts = np.unique(array, return_counts=True)
#    count_map = {val: count for val, count in zip(values, counts)}
#    sorted_array = sorted(array)
#    first = sorted_array[int(array.shape[0]/2) - 1]
#    second = sorted_array[int(array.shape[0]/2)]
#
#    if count_map[first] < count_map[second]:
#        return second
#
#    return first

def coalesce(val_a, val_b):
    """
    The coalesce functionality

    Args:
        val_a (obj/None): the first value
        val_b (obj/None): the second value

    Returns:
        obj: val_b if val_a is None otherwise val_a
    """
    if val_a is None:
        return val_b
    return val_a

# Difficulty with the potentially incoming pandas dataframes
#
#class AdjustedSimpleImputer(BaseEstimator, TransformerMixin):
#    def __init__(self, missing_values=np.nan):
#        self.missing_values = missing_values
#
#    def fit(self, input_array, y=None):
#        self.imputers = []
#        for idx in range(input_array.shape[1]):
#            adjusted_median_value = adjusted_median(input_array[:, idx])
#            self.imputers.append(SimpleImputer(missing_values=self.missing_values,
#                               strategy='constant',
#                               fill_value=adjusted_median_value).fit(input_array[:, [idx]]))
#
#        return self
#
#    def transform(self, input_array, y=None): # pylint: disable=invalid-name
#        output_array = input_array.copy()
#        _ = y
#
#        for idx in range(input_array.shape[1]):
#            output_array[:, idx] = self.imputers[idx].transform(input_array[:, [idx]])[:, 0]
#
#        return output_array
#
#    def get_feature_names_out(self, features):
#
#        return features

class IdentityTransformer(BaseEstimator, TransformerMixin):
    """
    Identity transformer
    """
    def __init__(self):
        """
        Constructor of the transformer
        """

    def fit(self, input_array, y=None): # pylint: disable=invalid-name
        """
        The fit function.

        Args:
            input_array (np.array): the input array
            y (np.array): the input labels

        Returns:
            obj: the fitted transformer
        """
        _ = input_array
        _ = y
        return self

    def transform(self, input_array, y=None): # pylint: disable=invalid-name
        """
        The transform function

        Args:
            input_array (np.array): the input array to transform
            y (np.array): the labels

        Returns:
            np.array: the transformed array
        """
        _ = y
        return input_array*1

    def get_feature_names_out(self, features):
        """
        Returns the feature names

        Args:
            features (list): the input feature names

        Returns:
            list: the output feature names
        """
        return features

class LabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label encoder
    """
    def __init__(self):
        """
        Constructor of the transformer
        """
        self.unique_mapping = None

    def fit(self, input_array, y=None): # pylint: disable=invalid-name
        """
        The fit function.

        Args:
            input_array (np.array): the input array
            y (np.array): the input labels

        Returns:
            obj: the fitted transformer
        """
        _ = y
        uniques = np.unique(input_array)
        self.unique_mapping = {item: idx for idx, item in enumerate(uniques)}
        return self

    def transform(self, input_array, y=None): # pylint: disable=invalid-name
        """
        The transform function

        Args:
            input_array (np.array): the input array to transform
            y (np.array): the labels

        Returns:
            np.array: the transformed array
        """
        _ = y
        return pd.DataFrame([self.unique_mapping[val[0]] for _, val in input_array.iterrows()])

    def get_feature_names_out(self, features):
        """
        Returns the feature names

        Args:
            features (list): the input feature names

        Returns:
            list: the output feature names
        """
        return features

def read_csv_data(filename,
                    *,
                    sep=',',
                    usecols=None,
                    header=None,
                    delim_whitespace=False,
                    decimal='.'):
    """
    Read a csv file

    Args:
        filename (str): path and filename
        sep (str): the separator
        usecols (list): the columns to use
        header (None/list): the header
        delim_whitespace (bool): delimeter whitespaces
        decimal (str): the decimal separator character

    Returns:
        pd.DataFrame: the read data
    """
    if delim_whitespace:
        return pd.read_csv(io.BytesIO(pkgutil.get_data('common_datasets', filename)),
                            header=header,
                            usecols=usecols,
                            delim_whitespace=delim_whitespace,
                            decimal=decimal)

    return pd.read_csv(io.BytesIO(pkgutil.get_data('common_datasets', filename)),
                        header=header,
                        usecols=usecols,
                        sep=sep,
                        decimal=decimal)

def read_xls_data(filename, sheet_name= None):
    """
    Read excel data

    Args:
        filename (str): path and filename
        sheet_name (str/None): the sheet name to read

    Returns:
        pd.DataFrame: the read data
    """
    if sheet_name is None:
        return pd.read_excel(io.BytesIO(pkgutil.get_data('common_datasets', filename)),
                                engine='openpyxl')

    return pd.read_excel(io.BytesIO(pkgutil.get_data('common_datasets', filename)),
                            sheet_name= sheet_name,
                            engine='openpyxl')

def read_arff_data(filename):
    """
    Read arff data

    Args:
        filename (str): path and filename

    Returns:
        np.array, obj: the data and the metadata
    """
    return arff.loadarff(io.StringIO(pkgutil.get_data('common_datasets',
                                                      filename).decode('unicode_escape')))

def determine_types(dataframe):
    """
    Determine the datatypes of csv columns

    Args:
        dataframe (pd.DataFrame): the dataframe to predict the datatypes of

    Returns:
        dict: the datatypes
    """
    tmp = pd.DataFrame(dataframe.nunique())

    tmp['type'] = 'category'
    tmp.loc[(dataframe.dtypes.apply(lambda x: np.issubdtype(x, np.number))), 'type'] = 'numeric'

    feature_types = tmp['type'].to_dict()

    return feature_types

def prepare_csv_data_template(dataset,
                                name,
                                target_label,
                                *,
                                feature_types=None,
                                problem_type='binary',
                                citation_key='krnn',
                                missing_data=None):
    """
    The processing and encoding of csv data

    Args:
        dataset (pd.DataFrame): the dataset to process
        name (str): the name of the dataset
        target_label (str): the target column
        feature_types (dict): te feature types
        problem_type (str): 'binary'/'multiclass'/'regression'
        citation_key (str): the citation key
        missing_data (dict): the missing data specification

    Returns:
        dict: the dataset in sklearn representation
    """
    dataset.columns= [str(col) for col in dataset.columns]

    feature_types = coalesce(feature_types, determine_types(dataset))

    dataprep = DataPreprocessor(dataset_raw=dataset,
                                target_label=target_label,
                                name=name,
                                feature_types=feature_types,
                                citation_key=citation_key,
                                problem_type=problem_type,
                                missing_data=missing_data)

    dataset = dataprep.get_dataset()

    return dataset

def prepare_xls_data_template(dataset,
                                name,
                                target_label,
                                *,
                                feature_types=None,
                                problem_type='regression',
                                citation_key='uci',
                                missing_data=None):
    """
    The processing and encoding of csv data

    Args:
        dataset (pd.DataFrame): the dataset to process
        name (str): the name of the dataset
        target_label (str): the target column
        feature_types (dict): the feature types
        citation_key (str): the citation key to use
        problem_type (str): 'binary'/'multiclass'/'regression'
        missing_data (dict): missing data specification

    Returns:
        dict: the dataset in sklearn representation
    """
    dataset.columns= [str(col) for col in dataset.columns]

    feature_types = coalesce(feature_types, determine_types(dataset))

    dataprep = DataPreprocessor(dataset_raw=dataset,
                                target_label=target_label,
                                name=name,
                                feature_types=feature_types,
                                citation_key=citation_key,
                                problem_type=problem_type,
                                missing_data=missing_data)

    dataset = dataprep.get_dataset()

    return dataset

def load_arff_template(path,
                        name,
                        target_label,
                        *,
                        feature_types=None,
                        citation_key='keel',
                        problem_type='binary',
                        missing_data=None):
    """
    Loading an arff dataset

    Args:
        path (str): the path of the data file
        name (str): the name of the dataset
        target_label (str): the label of the target column
        feature_types (dict): the feature types
        citation_key (str): the citation key to use
        problem_type (str): 'binary'/'multiclass'/'regression'
        missing_data (dict): missing data specification

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    dataset_raw, meta= read_arff_data(path)

    feature_types_derived = {attr: item.type_name
                                for attr, item in meta._attributes.items()} # pylint: disable=protected-access
    feature_types = coalesce(feature_types,
                             feature_types_derived)

    dataset_raw = pd.DataFrame(dataset_raw,
                                columns=list(feature_types.keys()))

    dataprep = DataPreprocessor(dataset_raw,
                            feature_types=feature_types,
                            target_label=target_label,
                            name=name,
                            citation_key=citation_key,
                            problem_type=problem_type,
                            missing_data=missing_data)

    dataset = dataprep.get_dataset()

    return dataset

def load_arff_template_binary(path,
                                name,
                                target_label,
                                *,
                                citation_key='keel',
                                feature_types=None,
                                missing_data=None):
    """
    Loading an arff dataset

    Args:
        path (str): the path of the data file
        name (str): the name of the dataset
        target_label (str): the label of the target column
        citation_key (str): the citation key to use
        rever_target (bool): whether to revert the target label
        feature_types (dict): the feature types
        missing_data (dict): the missing data specification

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    return load_arff_template(path=path,
                                name=name,
                                target_label=target_label,
                                feature_types=feature_types,
                                citation_key=citation_key,
                                problem_type='binary',
                                missing_data=missing_data)

def load_arff_template_multiclass(path,
                                    name,
                                    target_label,
                                    *,
                                    citation_key='keel',
                                    feature_types=None,
                                    missing_data=None):
    """
    Loading an arff dataset

    Args:
        path (str): the path of the data file
        name (str): the name of the dataset
        target_label (str): the label of the target column
        citation_key (str): the citation key to use
        feature_types (dict): the feature types
        missing_data (dict): the missing data specification

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    return load_arff_template(path=path,
                                name=name,
                                target_label=target_label,
                                feature_types=feature_types,
                                citation_key=citation_key,
                                problem_type='multiclass',
                                missing_data=missing_data)

def load_arff_template_regression(path,
                                    name,
                                    target_label,
                                    *,
                                    citation_key='keel',
                                    feature_types=None,
                                    missing_data=None):
    """
    Loading an arff dataset

    Args:
        path (str): the path of the data file
        name (str): the name of the dataset
        target_label (str): the label of the target column
        citation_key (str): the citation key to use
        feature_types (dict): the feature types
        missing_data (dict): the missing data specification

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    return load_arff_template(path=path,
                                name=name,
                                target_label=target_label,
                                feature_types=feature_types,
                                citation_key=citation_key,
                                problem_type='regression',
                                missing_data=missing_data)

def numeric_preprocessing(missing_values=np.nan, strategy='median'):
    """
    Pipeline for processing numeric features

    Args:
        missing_values (obj/list/value): the missing value indicator
        strategy (str): the imputation strategy

    Returns:
        Transformer: the pipeline
    """
    simple_imputer = SimpleImputer(missing_values=missing_values,
                                   strategy=strategy)
    #simple_imputer = AdjustedSimpleImputer(missing_values=missing_values)
    #missing_indicator = MissingIndicator(missing_values=missing_values)

    #feature_union = FeatureUnion([('imputer', simple_imputer),
    #                              ('missing_indicator', missing_indicator)],
    #                              n_jobs=1)
    feature_union = FeatureUnion([('imputer', simple_imputer)],
                                  n_jobs=1)

    return feature_union

def numeric_grid_preprocessing(missing_values=np.nan, strategy='most_frequent'):
    """
    Pipeline for processing numeric features

    Args:
        missing_values (obj/list/value): the missing value indicator
        strategy (str): the imputation strategy

    Returns:
        Transformer: the pipeline
    """
    simple_imputer = SimpleImputer(missing_values=missing_values,
                                   strategy=strategy)
    #simple_imputer = KNNImputer(missing_values=missing_values, n_neighbors=1)
    #simple_imputer = AdjustedSimpleImputer(missing_values=missing_values)
    #missing_indicator = MissingIndicator(missing_values=missing_values)

    #feature_union = FeatureUnion([('imputer', simple_imputer),
    #                              ('missing_indicator', missing_indicator)],
    #                              n_jobs=1)
    feature_union = FeatureUnion([('imputer', simple_imputer)],
                                  n_jobs=1)

    return feature_union

def category_preprocessing(missing_values='?',
                           strategy='most_frequent'):
    """
    Pipeline for processing category features

    Args:
        missing_values (obj/list/value): the missing value indicator
        strategy (str): the imputation strategy

    Returns:
        Transformer: the pipeline
    """
    simple_imputer = SimpleImputer(missing_values=missing_values,
                                   strategy=strategy)
    #missing_indicator = MissingIndicator(missing_values=missing_values)

    encoding = OneHotEncoder(drop='first', sparse_output=False)

    pipeline = Pipeline([('imputer', simple_imputer),
                         ('encoding', encoding)])

    #feature_union = FeatureUnion([('imputation_encoding', pipeline),
    #                              ('missing_indicator', missing_indicator)])
    feature_union = FeatureUnion([('imputation_encoding', pipeline)])

    return feature_union

def ordinal_preprocessing(missing_values='?',
                            strategy='most_frequent'):
    """
    Pipeline for processing ordinal features

    Args:
        missing_values (obj/list/value): the missing value indicator
        strategy (str): the imputation strategy

    Returns:
        Transformer: the pipeline
    """
    simple_imputer = SimpleImputer(missing_values=missing_values,
                                   strategy=strategy)
    #missing_indicator = MissingIndicator(missing_values=missing_values)

    encoding = OrdinalEncoder()

    pipeline = Pipeline([('imputer', simple_imputer),
                         ('encoding', encoding)])

    #feature_union = FeatureUnion([('imputation_encoding', pipeline),
    #                              ('missing_indicator', missing_indicator)])
    feature_union = FeatureUnion([('imputation_encoding', pipeline)])

    return feature_union

def class_label_preprocessing():
    """
    Pipeline for processing binary class labels

    Returns:
        Pipeline: the pipeline
    """
    encoding = OneHotEncoder(drop='first', sparse_output=False)

    return Pipeline([('class_label', encoding)])

def multiclass_label_preprocessing():
    """
    Pipeline for processing multiclass class labels

    Returns:
        Pipeline: the pipeline
    """
    encoding = LabelEncoder()

    return Pipeline([('multiclass_label', encoding)])

class DataPreprocessor:
    """
    The data preprocessor class
    """
    def __init__(self,
                 dataset_raw,
                 *,
                 grid_threshold='sqrt',
                 target_label=None,
                 missing_data=None,
                 name=None,
                 feature_types=None,
                 citation_key=None,
                 problem_type='binary'):
        """
        The constructor

        Args:
            dataset_raw (pd.DataFrame): the raw dataset
            target_label (str): the target label
            missing_data (dict): the missing data specification
            name (str): the name of the dataset
            feature_types (dict): the feature types
            citation_key (str): the citation key
            problem_type (str): 'binary'/'multiclass'/'regression'
        """

        if missing_data is None:
            missing_data = {'numeric': {'strategy': 'median', 'missing_values': np.nan},
                            'numeric_grid': {'strategy': 'most_frequent', 'missing_values': np.nan},
                            'category': {'strategy': 'most_frequent', 'missing_values': '?'},
                            'ordinal': {'strategy': 'most_frequent', 'missing_values': '?'}}

        self.descriptor = {'name': name,
                           'target_label': target_label,
                           'problem_type': problem_type,
                           'citation_key': citation_key}
        self.grid_threshold = grid_threshold
        self.missing_data = missing_data

        self.feature_types = feature_types

        self.dataset_raw = dataset_raw

        self.grid = None

    def category_features(self):
        """
        Category features

        Returns:
            list: the category features
        """
        return [key for key, item in self.feature_types.items()
                    if item in ['nominal', 'category']
                        and key != self.descriptor['target_label']]

    def numeric_features(self):
        """
        Numeric features

        Returns:
            list: the list of numeric features
        """
        assert len(self.feature_types) == len(self.grid)

        return [key for idx, (key, item) in enumerate(self.feature_types.items())
                    if item in ['numeric', 'real', 'integer']
                        and key != self.descriptor['target_label'] and not self.grid[idx]]

    def numeric_grid_features(self):
        """
        Numeric features

        Returns:
            list: the list of numeric features
        """
        assert len(self.feature_types) == len(self.grid)

        return [key for idx, (key, item) in enumerate(self.feature_types.items())
                    if item in ['numeric', 'real', 'integer']
                        and key != self.descriptor['target_label'] and self.grid[idx]]

    def ordinal_features(self):
        """
        Ordinal features

        Returns:
            list: the list of ordinal features
        """
        return [key for key, item in self.feature_types.items()
                    if item in ['ordinal'] and key != self.descriptor['target_label']]

    def _transform_dataset(self):
        """
        Carry out the imputation and encoding

        Returns:
            pd.DataFrame: the transformed dataset
        """
        numerical = numeric_preprocessing(**self.missing_data['numeric'])
        numerical_grid = numeric_grid_preprocessing(**self.missing_data['numeric_grid'])
        categorical = category_preprocessing(**self.missing_data['category'])
        ordinal = ordinal_preprocessing(**self.missing_data['ordinal'])

        if self.descriptor['problem_type'] == 'binary':
            target_label = class_label_preprocessing()
        elif self.descriptor['problem_type'] == 'multiclass':
            target_label = multiclass_label_preprocessing()
        elif self.descriptor['problem_type'] == 'regression':
            target_label = IdentityTransformer()

        n_features = len(self.numeric_features()) + len(self.category_features())\
                        + len(self.numeric_grid_features()) + len(self.ordinal_features()) + 1
        message = f"{str(n_features)} {len(self.dataset_raw.columns)}"
        assert n_features == len(self.dataset_raw.columns), message

        feature_union = ColumnTransformer([('numerical', numerical, self.numeric_features()),
                                ('numerical_grid', numerical_grid, self.numeric_grid_features()),
                                ('category', categorical, self.category_features()),
                                ('ordinal', ordinal, self.ordinal_features()),
                                ('target_label', target_label, [self.descriptor['target_label']])],
                                n_jobs=1)

        feature_union.fit(self.dataset_raw)

        transformed = feature_union.transform(self.dataset_raw)
        names = feature_union.get_feature_names_out()

        pdf = pd.DataFrame(transformed, columns=names)
        pdf = pdf.rename({names[-1]: self.descriptor['target_label']}, axis='columns')

        total_target = np.sum(pdf[self.descriptor['target_label']]) # pylint: disable=unsubscriptable-object
        total_opposite = np.sum(1 - pdf[self.descriptor['target_label']]) # pylint: disable=unsubscriptable-object

        if (self.descriptor['problem_type'] == 'binary' and total_target > total_opposite):
            pdf[self.descriptor['target_label']] = 1 - pdf[self.descriptor['target_label']] # pylint: disable=unsubscriptable-object,unsupported-assignment-operation

        return pdf

    def dataset_phenotype(self, dataset):
        """
        Determine the dataset phenotype

        Args:
            dataset (str): the dataset name

        Returns:
            str: the dataset phenotype
        """
        dataset = dataset.replace('_', ' ').replace('-', ' ').split(' ')[0]
        dataset = dataset.rstrip('0123456789')

        return dataset

    def get_dataset(self):
        """
        Get the dataset in sklearn.datasets representation

        Returns:
            dict: the dataset in sklearn.datasets representation
        """
        result = {}

        grid_function = np.sqrt
        if self.grid_threshold == 'log2':
            grid_function = np.log2

        self.grid = []
        for _, column in enumerate(self.dataset_raw.columns):
            if pd.api.types.is_numeric_dtype(self.dataset_raw[column]):
                array = self.dataset_raw[column].values
                diffs = np.diff(sorted(np.unique(array)), 1)
                self.grid.append(bool(len(np.unique(diffs)) < grid_function(len(np.unique(array)))))
            else:
                self.grid.append(False)

        transformed = self._transform_dataset()

        result['data'] = transformed.drop(self.descriptor['target_label'],
                                          axis='columns').values.astype(float)
        if self.descriptor['problem_type'] in ['binary', 'multiclass']:
            result['target'] = transformed[self.descriptor['target_label']].values.astype(int) # pylint: disable=unsubscriptable-object
            result['mutual_information'] = mutual_info_classif(result['data'],
                                                               result['target']).tolist()
        elif self.descriptor['problem_type'] == 'regression':
            result['target'] = transformed[self.descriptor['target_label']].values.astype(float) # pylint: disable=unsubscriptable-object
            result['mutual_information'] = mutual_info_regression(result['data'],
                                                                  result['target']).tolist()

        X = result['data']

        result['grid'] = self.grid
        result['n_feature_uniques'] = [len(np.unique(X[:, idx])) for idx in range(X.shape[1])]

        result['feature_names'] = list(transformed.columns[:-1])
        result['feature_types'] = self.feature_types
        result['target_label'] = self.descriptor['target_label']
        result['name'] = self.descriptor['name']
        result['phenotype'] = self.dataset_phenotype(self.descriptor['name'])
        result['citation'] = references.get(self.descriptor['citation_key'], None)
        result['citation_key'] = self.descriptor['citation_key']
        result['n_col'] = len(result['feature_names'])
        result['n_col_orig'] = len(self.dataset_raw.columns) - 1
        result['n_col_non_unique_orig'] = np.sum(self.dataset_raw.nunique() > 1) - 1
        result['n'] = len(result['target'])
        result['DESCR'] = self.descriptor['name']

        if self.descriptor['problem_type'] == 'binary':
            result['n_minority'] = np.sum(result['target'] == 1)
            imb_ratio = np.sum(result['target'] == 0) / np.sum(result['target'] == 1)
            result['imbalance_ratio'] = imb_ratio
        if self.descriptor['problem_type'] == 'multiclass':
            result['n_classes'] = len(np.unique(result['target']))

        return result
