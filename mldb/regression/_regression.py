import numpy as np
import pandas as pd

from mldb._io import read_csv_data, read_xls_data, read_arff_data, construct_return_set, citations

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
          'load_ele_1',
          'load_treasury',
          'load_compactiv',
          'load_puma32h',
          'summary',
          'get_filtered_data_loaders',
          'get_data_loaders']

def load_airfoil(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/regression/airfoil/airfoil_self_noise.dat.txt', sep= '\t')
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "airfoil", return_X_y, encode, citation= citations['uci'], name= "airfoil", verbose= verbose, problem_type='regression')

def load_cpu_performance(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/regression/cpu_performance/machine.data.txt', sep= ',')
    del db[db.columns[-1]]
    del db[db.columns[1]]
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "cpu_performance", return_X_y, encode, citation= citations['uci'], name= "cpu_performance", verbose= verbose, problem_type='regression')

def load_forestfires(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/regression/forestfires/forestfires.csv', sep= ',', header=0)
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "forestfires", return_X_y, encode, citation= citations['uci'], name= "forestfires", verbose= verbose, problem_type='regression')

def load_real_estate_valuation(return_X_y= False, encode= True, verbose= False):
    db= read_xls_data('data/regression/real_estate_valuation/Real estate valuation data set.xlsx')
    del db[db.columns[0]]
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    for c in db.columns:
        db[c]= db[c].astype(float)
    
    return construct_return_set(db, "real_estate_valuation", return_X_y, encode, citation= citations['uci'], name= "real_estate_valuation", verbose= verbose, problem_type='regression')

def load_residential_building(return_X_y= False, encode= True, verbose= False):
    # target: V9
    
    db= read_xls_data('data/regression/residential_building/Residential-Building-Data-Set.xlsx')
    db= db.drop(0, axis='index')
    db.reset_index(drop=True, inplace=True)
    del db[db.columns[-1]]
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    for c in db.columns:
        db[c]= db[c].astype(float)
    
    return construct_return_set(db, "residential_building", return_X_y, encode, citation= citations['uci'], name= "residential_building", verbose= verbose, problem_type='regression')

def load_slump_test(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/regression/slump_test/slump_test.data.txt', sep= ',', header=0)
    del db[db.columns[0]]
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "slump_test", return_X_y, encode, citation= citations['uci'], name= "slump_test", verbose= verbose, problem_type='regression')

def load_stock_portfolio_performance(return_X_y= False, encode= True, verbose= False):
    # target: normalized annual return
    
    db= read_xls_data('data/regression/stock_portfolio_performance/stock portfolio performance data set.xlsx', sheet_name='all period')
    db.columns= db.iloc[0].values
    db= db.drop(db.index[0], axis='index')
    db.reset_index(drop=True, inplace=True)
    del db[db.columns[0]]
    columns= list(db.columns[0:5]) + [db.columns[11]]
    db= db[columns]
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    for c in db.columns:
        db[c]= db[c].astype(float)
    
    return construct_return_set(db, "stock_portfolio_performance", return_X_y, encode, citation= citations['uci'], name= "stock_portfolio_performance", verbose= verbose, problem_type='regression')

def load_winequality_red(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/regression/winequality_red/winequality-red.csv', sep= ';', header=0)
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "winequality_red", return_X_y, encode, citation= citations['uci'], name= "winequality_red", verbose= verbose, problem_type='regression')

def load_winequality_white(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/regression/winequality_white/winequality-white.csv', sep= ';', header=0)
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "winequality_white", return_X_y, encode, citation= citations['uci'], name= "winequality_white", verbose= verbose, problem_type='regression')

def load_yacht_hydrodynamics(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/regression/yacht_hydrodynamics/yacht_hydrodynamics.data.txt', sep= None, header=0, delim_whitespace=True)
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "yacht_hydrodynamics", return_X_y, encode, citation= citations['uci'], name= "yacht_hydrodynamics", verbose= verbose, problem_type='regression')

def load_ccpp(return_X_y= False, encode= True, verbose= False):
    db= read_xls_data('data/regression/ccpp/Folds5x2_pp.xlsx', sheet_name='Sheet1')
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "ccpp", return_X_y, encode, citation= citations['uci'], name= "ccpp", verbose= verbose, problem_type='regression')

def load_communities(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/regression/communities/communities.data', sep= ',')
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "communities", return_X_y, encode, citation= citations['uci'], name= "communities", verbose= verbose, problem_type='regression')

def load_diabetes(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/diabetes/diabetes.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "diabetes", return_X_y, encode, citation= citations['keel'], name= "diabetes", verbose= verbose, problem_type= 'regression')

def load_laser(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/laser/laser.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "laser", return_X_y, encode, citation= citations['keel'], name= "laser", verbose= verbose, problem_type= 'regression')

def load_autoMPG6(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/autoMPG6/autoMPG6.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "autoMPG6", return_X_y, encode, citation= citations['keel'], name= "autoMPG6", verbose= verbose, problem_type= 'regression')

def load_wizmir(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/wizmir/wizmir.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wizmir", return_X_y, encode, citation= citations['keel'], name= "wizmir", verbose= verbose, problem_type= 'regression')

def load_wankara(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/wankara/wankara.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wankara", return_X_y, encode, citation= citations['keel'], name= "wankara", verbose= verbose, problem_type= 'regression')

def load_mortgage(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/mortgage/mortgage.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "mortgage", return_X_y, encode, citation= citations['keel'], name= "mortgage", verbose= verbose, problem_type= 'regression')

def load_baseball(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/baseball/baseball.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "baseball", return_X_y, encode, citation= citations['keel'], name= "baseball", verbose= verbose, problem_type= 'regression')

def load_ele_1(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/ele_1/ele-1.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ele_1", return_X_y, encode, citation= citations['keel'], name= "ele_1", verbose= verbose, problem_type= 'regression')

def load_treasury(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/treasury/treasury.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "treasury", return_X_y, encode, citation= citations['keel'], name= "treasury", verbose= verbose, problem_type= 'regression')

def load_compactiv(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/compactiv/compactiv.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "compactiv", return_X_y, encode, citation= citations['keel'], name= "compactiv", verbose= verbose, problem_type= 'regression')

def load_puma32h(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/regression/puma32h/puma32h.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "puma32h", return_X_y, encode, citation= citations['keel'], name= "puma32h", verbose= verbose, problem_type= 'regression')

def summary(include_citation= True, subset= 'study'):
    results= []
    # fixing the globals dictionary keys
    
    for func_name in __all__:
        if func_name.startswith('load_') and not func_name.startswith('load_forest'):
            data_not_encoded= globals()[func_name](return_X_y= False, encode= False)
            data_encoded= globals()[func_name](return_X_y= False, encode= True)
            
            result= {'loader_function': globals()[func_name],
                            'name': data_not_encoded['name'],
                            'len': len(data_not_encoded['data']),
                            'non_encoded_n_attr': len(data_not_encoded['data'][0]),
                            'encoded_n_attr': len(data_encoded['data'][0])}
            
            if include_citation:
                result['citation']= data_encoded['citation']
            
            results.append(result)
    
    df_results= pd.DataFrame(results)

    return df_results

def get_filtered_data_loaders(num_features_lower_bound= 1,
                              num_features_upper_bound= 5000,
                              len_lower_bound= 1,
                              len_upper_bound= 10000):
    descriptors= summary()
    return descriptors[(descriptors['len'] >= len_lower_bound) & 
                       (descriptors['len'] < len_upper_bound) & 
                       (descriptors['encoded_n_attr'] >= num_features_lower_bound) & 
                       (descriptors['encoded_n_attr'] < num_features_upper_bound)]['loader_function'].values

def get_data_loaders(subset='all'):
    """
    Args:
        subset (str): 'all'/'study'/'small'/'tiny'
    """
    
    num_features_lower_bound= 1
    len_lower_bound= 1
    num_features_upper_bound= 1e10
    len_upper_bound= 1e10
    
    if subset == 'study':
        num_features_upper_bound= 100
        len_upper_bound= 1000
    elif subset == 'small':
        num_features_upper_bound= 100
        len_upper_bound= 10000
    elif subset == 'tiny':
        len_upper_bound= 120
    
    return get_filtered_data_loaders(num_features_lower_bound,
                                     num_features_upper_bound,
                                     len_lower_bound,
                                     len_upper_bound)
