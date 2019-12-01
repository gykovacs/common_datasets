import numpy as np
import pandas as pd

from mldb._io import read_csv_data, read_xls_data, read_arff_data, construct_return_set, references

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
          'generate_summary_table',
          'get_filtered_data_loaders',
          'get_data_loaders',
          'get_references']

def get_references():
    return references

def load_airfoil(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_csv_data('data/regression/airfoil/airfoil_self_noise.dat.txt', sep= '\t')
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "airfoil", return_X_y, encode, citation= 'uci', name= "airfoil", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_cpu_performance(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_csv_data('data/regression/cpu_performance/machine.data.txt', sep= ',')
    del db[db.columns[-1]]
    del db[db.columns[1]]
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "cpu_performance", return_X_y, encode, citation= 'uci', name= "cpu_performance", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_forestfires(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_csv_data('data/regression/forestfires/forestfires.csv', sep= ',', header=0)
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "forestfires", return_X_y, encode, citation= 'uci', name= "forestfires", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_real_estate_valuation(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_xls_data('data/regression/real_estate_valuation/Real estate valuation data set.xlsx')
    del db[db.columns[0]]
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    for c in db.columns:
        db[c]= db[c].astype(float)
    
    return construct_return_set(db, "real_estate_valuation", return_X_y, encode, citation= 'uci', name= "real_estate_valuation", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_residential_building(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
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
    
    return construct_return_set(db, "residential_building", return_X_y, encode, citation= 'uci', name= "residential_building", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_slump_test(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_csv_data('data/regression/slump_test/slump_test.data.txt', sep= ',', header=0)
    del db[db.columns[0]]
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "slump_test", return_X_y, encode, citation= 'uci', name= "slump_test", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_stock_portfolio_performance(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
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
    
    return construct_return_set(db, "stock_portfolio_performance", return_X_y, encode, citation= 'uci', name= "stock_portfolio_performance", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_winequality_red(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_csv_data('data/regression/winequality_red/winequality-red.csv', sep= ';', header=0)
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "winequality_red", return_X_y, encode, citation= 'uci', name= "winequality_red", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_winequality_white(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_csv_data('data/regression/winequality_white/winequality-white.csv', sep= ';', header=0)
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "winequality_white", return_X_y, encode, citation= 'uci', name= "winequality_white", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_yacht_hydrodynamics(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_csv_data('data/regression/yacht_hydrodynamics/yacht_hydrodynamics.data.txt', sep= None, header=0, delim_whitespace=True)
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "yacht_hydrodynamics", return_X_y, encode, citation= 'uci', name= "yacht_hydrodynamics", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_ccpp(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_xls_data('data/regression/ccpp/Folds5x2_pp.xlsx', sheet_name='Sheet1')
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "ccpp", return_X_y, encode, citation= 'uci', name= "ccpp", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_communities(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    db= read_csv_data('data/regression/communities/communities.data', sep= ',')
    columns= list(db.columns)
    columns[-1]= 'target'
    db.columns= columns
    
    return construct_return_set(db, "communities", return_X_y, encode, citation= 'uci', name= "communities", verbose= verbose, problem_type='regression', onehot_threshold=onehot_threshold)

def load_diabetes(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/diabetes/diabetes.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "diabetes", return_X_y, encode, citation= 'keel', name= "diabetes", verbose= verbose, problem_type= 'regression')

def load_laser(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/laser/laser.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "laser", return_X_y, encode, citation= 'keel', name= "laser", verbose= verbose, problem_type= 'regression')

def load_autoMPG6(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/autoMPG6/autoMPG6.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "autoMPG6", return_X_y, encode, citation= 'keel', name= "autoMPG6", verbose= verbose, problem_type= 'regression')

def load_wizmir(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/wizmir/wizmir.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wizmir", return_X_y, encode, citation= 'keel', name= "wizmir", verbose= verbose, problem_type= 'regression')

def load_wankara(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/wankara/wankara.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "wankara", return_X_y, encode, citation= 'keel', name= "wankara", verbose= verbose, problem_type= 'regression')

def load_mortgage(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/mortgage/mortgage.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "mortgage", return_X_y, encode, citation= 'keel', name= "mortgage", verbose= verbose, problem_type= 'regression')

def load_baseball(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/baseball/baseball.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "baseball", return_X_y, encode, citation= 'keel', name= "baseball", verbose= verbose, problem_type= 'regression')

def load_ele_1(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/ele_1/ele-1.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "ele_1", return_X_y, encode, citation= 'keel', name= "ele_1", verbose= verbose, problem_type= 'regression')

def load_treasury(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/treasury/treasury.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "treasury", return_X_y, encode, citation= 'keel', name= "treasury", verbose= verbose, problem_type= 'regression')

def load_compactiv(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/compactiv/compactiv.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "compactiv", return_X_y, encode, citation= 'keel', name= "compactiv", verbose= verbose, problem_type= 'regression')

def load_puma32h(return_X_y= False, encode= True, verbose= False, onehot_threshold= 10):
    data, meta= read_arff_data('data/regression/puma32h/puma32h.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "puma32h", return_X_y, encode, citation= 'keel', name= "puma32h", verbose= verbose, problem_type= 'regression')

def generate_summary():
    results= []
    
    for func_name in __all__:
        if func_name.startswith('load_'):
            data_encoded= globals()[func_name](return_X_y= False, encode= True)
            
            result= {'loader_function': globals()[func_name],
                        'name': data_encoded['name'],
                        'n': len(data_encoded['data']),
                        'n_attr_raw': len(data_encoded['data_raw'][0]),
                        'n_attr_encoded': len(data_encoded['data'][0])}
            
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
                              n_bounds= [1, 10000]):
    descriptors= summary()
    return descriptors[(descriptors['n'] >= n_bounds[0]) & (descriptors['n'] < n_bounds[1]) & 
                       (descriptors['n_attr_encoded'] >= n_attr_encoded_bounds[0]) & (descriptors['n_attr_encoded'] < n_attr_encoded_bounds[1]) & 
                       (descriptors['n_attr_raw'] >= n_attr_raw_bounds[0]) & (descriptors['n_attr_raw'] < n_attr_raw_bounds[1])]['loader_function'].values

def get_data_loaders(subset='all'):
    """
    Args:
        subset (str): 'all'/'study'/'small'/'tiny'
    """
    
    n_attr_encoded_bounds= [1, 5000]
    n_attr_raw_bounds= [1, 5000]
    n_bounds= [1, 10000]
    
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
                                    n_bounds= n_bounds)

summary_table= np.array([
       ['load_airfoil', 1503, 5, 5, 'airfoil', 'uci'],
       ['load_cpu_performance', 209, 7, 7, 'cpu_performance', 'uci'],
       ['load_forestfires', 517, 18, 12, 'forestfires', 'uci'],
       ['load_real_estate_valuation', 414, 6, 6, 'real_estate_valuation', 'uci'],
       ['load_residential_building', 372, 107, 107, 'residential_building', 'uci'],
       ['load_slump_test', 103, 9, 9, 'slump_test', 'uci'],
       ['load_stock_portfolio_performance', 63, 6, 6, 'stock_portfolio_performance', 'uci'],
       ['load_winequality_red', 1599, 11, 11, 'winequality_red', 'uci'],
       ['load_winequality_white', 4898, 11, 11, 'winequality_white', 'uci'],
       ['load_yacht_hydrodynamics', 307, 6, 6, 'yacht_hydrodynamics', 'uci'],
       ['load_ccpp', 9568, 4, 4, 'ccpp', 'uci'],
       ['load_communities', 1994, 154, 127, 'communities', 'uci'],
       ['load_diabetes', 43, 2, 2, 'diabetes', 'keel'],
       ['load_laser', 993, 4, 4, 'laser', 'keel'],
       ['load_autoMPG6', 392, 5, 5, 'autoMPG6', 'keel'],
       ['load_wizmir', 1461, 9, 9, 'wizmir', 'keel'],
       ['load_wankara', 321, 9, 9, 'wankara', 'keel'],
       ['load_mortgage', 1049, 15, 15, 'mortgage', 'keel'],
       ['load_baseball', 337, 16, 16, 'baseball', 'keel'],
       ['load_ele_1', 495, 2, 2, 'ele_1', 'keel'],
       ['load_treasury', 1049, 15, 15, 'treasury', 'keel'],
       ['load_compactiv', 8192, 21, 21, 'compactiv', 'keel'],
       ['load_puma32h', 8192, 32, 32, 'puma32h', 'keel']], dtype=object)

summary_columns= ['loader_function', 'n', 'n_attr_encoded', 'n_attr_raw', 'name',
       'reference_key']