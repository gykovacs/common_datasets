import numpy as np
import pandas as pd

from mldb._io import read_csv_data, read_arff_data, construct_return_set, references

__all__= ['load_glass',
          'load_satimage',
          'load_ecoli',
          'load_abalone',
          'load_yeast',
          'load_winequality_red',
          'load_winequality_white',
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
          'summary',
          'generate_summary_table',
          'get_filtered_data_loaders',
          'get_data_loaders',
          'get_references']

def get_references():
    return references

def load_glass(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    db= read_csv_data('data/classification/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "glass", return_X_y, encode, citation= 'krnn', name= "glass", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_satimage(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    db0= read_csv_data('data/classification/satimage/sat.trn.txt', sep= ' ')
    db1= read_csv_data('data/classification/satimage/sat.tst.txt', sep= ' ')
    db= pd.concat([db0, db1])
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "SATIMAGE", return_X_y, encode, citation= 'krnn', name= "SATIMAGE", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_ecoli(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    db= read_csv_data('data/classification/ecoli/ecoli.data.txt', delim_whitespace=True)
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    print(db.columns)
    return construct_return_set(db, "ecoli", return_X_y, encode, citation= 'krnn', name= "ecoli", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_abalone(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    db= read_csv_data('data/classification/abalone/abalone.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "abalone", return_X_y, encode, citation= 'krnn', name= "abalone", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_yeast(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    db= read_csv_data('data/classification/yeast/yeast.data.txt', delim_whitespace=True)
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "yeast", return_X_y, encode, citation= 'krnn', name= "yeast", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_winequality_red(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    db= read_csv_data('data/classification/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "winequality_red", return_X_y, encode, citation= 'krnn', name= "winequality_red", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_winequality_white(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    db= read_csv_data('data/classification/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "winequality_white", return_X_y, encode, citation= 'krnn', name= "winequality_white", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_automobile(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/automobile/automobile.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "automobile", return_X_y, encode, citation= 'keel', name= "automobile", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_balance(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/balance/balance.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "balance", return_X_y, encode, citation= 'keel', name= "balance", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_car(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/car/car.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "car", return_X_y, encode, citation= 'keel', name= "car", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_cleveland(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/cleveland/cleveland.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "cleveland", return_X_y, encode, citation= 'keel', name= "cleveland", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_contraceptive(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/contraceptive/contraceptive.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "contraceptive", return_X_y, encode, citation= 'keel', name= "contraceptive", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_dermatology(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/dermatology/dermatology.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "dermatology", return_X_y, encode, citation= 'keel', name= "dermatology", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_fars(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/fars/fars.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "fars", return_X_y, encode, citation= 'keel', name= "fars", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_flare(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/flare/flare.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "flare", return_X_y, encode, citation= 'keel', name= "flare", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_hayes_roth(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/hayes-roth/hayes-roth.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "hayes_roth", return_X_y, encode, citation= 'keel', name= "hayes_roth", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_kr_vs_k(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/kr-vs-k/kr-vs-k.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr_vs_k", return_X_y, encode, citation= 'keel', name= "kr_vs_k", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_led7digit(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/led7digit/led7digit.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "led7digit", return_X_y, encode, citation= 'keel', name= "led7digit", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_movement_libras(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/movement_libras/movement_libras.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "movement_libras", return_X_y, encode, citation= 'keel', name= "movement_libras", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_newthyroid(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/newthyroid/newthyroid.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "newthyroid", return_X_y, encode, citation= 'keel', name= "newthyroid", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_nursery(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/nursery/nursery.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "nursery", return_X_y, encode, citation= 'keel', name= "nursery", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_page_blocks(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/page-blocks/page-blocks.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "page_blocks", return_X_y, encode, citation= 'keel', name= "page_blocks", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_post_operative(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/post-operative/post-operative.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "post_operative", return_X_y, encode, citation= 'keel', name= "post_operative", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_segment(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/segment/segment.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "segment", return_X_y, encode, citation= 'keel', name= "segment", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_splice(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/splice/splice.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "splice", return_X_y, encode, citation= 'keel', name= "splice", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_tae(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/tae/tae.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "tae", return_X_y, encode, citation= 'keel', name= "tae", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_vowel(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/vowel/vowel.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vowel", return_X_y, encode, citation= 'keel', name= "vowel", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def load_zoo(return_X_y= False, encode= True, verbose= False, onehot_threshold=10):
    data, meta= read_arff_data('data/classification/zoo/zoo.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "zoo", return_X_y, encode, citation= 'keel', name= "zoo", verbose= verbose, problem_type='classification', onehot_threshold=onehot_threshold)

def generate_summary():
    results= []
    
    for func_name in __all__:
        if func_name.startswith('load_'):
            data_encoded= globals()[func_name](return_X_y= False, encode= True)
            
            result= {'loader_function': globals()[func_name],
                        'name': data_encoded['name'],
                        'n': len(data_encoded['data']),
                        'n_attr_raw': len(data_encoded['data_raw'][0]),
                        'n_attr_encoded': len(data_encoded['data'][0]),
                        'n_classes': len(np.unique(data_encoded['target']))}
            
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
                              n_classes_bounds= [1, 100]):
    descriptors= summary()
    return descriptors[(descriptors['n'] >= n_bounds[0]) & (descriptors['n'] < n_bounds[1]) & 
                       (descriptors['n_attr_encoded'] >= n_attr_encoded_bounds[0]) & (descriptors['n_attr_encoded'] < n_attr_encoded_bounds[1]) & 
                       (descriptors['n_attr_raw'] >= n_attr_raw_bounds[0]) & (descriptors['n_attr_raw'] < n_attr_raw_bounds[1]) &
                       (descriptors['n_classes'] >= n_classes_bounds[0]) & (descriptors['n_classes'] < n_classes_bounds[1])]['loader_function'].values

def get_data_loaders(subset='all'):
    """
    Args:
        subset (str): 'all'/'study'/'small'/'tiny'
    """
    
    n_attr_encoded_bounds= [1, 5000]
    n_attr_raw_bounds= [1, 5000]
    n_bounds= [1, 10000]
    n_classes_bounds= [1, 100]
    
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
                                    n_classes_bounds= n_classes_bounds)

summary_table=np.array([['load_glass', 214, 9, 9, 6, 'glass', 'krnn'],
       ['load_satimage', 6435, 36, 36, 6, 'SATIMAGE', 'krnn'],
       ['load_ecoli', 336, 7, 7, 8, 'ecoli', 'krnn'],
       ['load_abalone', 4177, 7, 7, 28, 'abalone', 'krnn'],
       ['load_yeast', 1484, 8, 8, 10, 'yeast', 'krnn'],
       ['load_winequality_red', 214, 9, 9, 6, 'winequality_red', 'krnn'],
       ['load_winequality_white', 214, 9, 9, 6, 'winequality_white', 'krnn'],
       ['load_automobile', 159, 43, 25, 6, 'automobile', 'keel'],
       ['load_balance', 625, 4, 4, 3, 'balance', 'keel'],
       ['load_car', 1728, 21, 6, 4, 'car', 'keel'],
       ['load_cleveland', 297, 13, 13, 5, 'cleveland', 'keel'],
       ['load_contraceptive', 1473, 9, 9, 3, 'contraceptive', 'keel'],
       ['load_dermatology', 358, 34, 34, 6, 'dermatology', 'keel'],
       ['load_fars', 100968, 107, 29, 8, 'fars', 'keel'],
       ['load_flare', 1066, 37, 11, 6, 'flare', 'keel'],
       ['load_hayes_roth', 160, 4, 4, 3, 'hayes_roth', 'keel'],
       ['load_kr_vs_k', 28056, 40, 6, 18, 'kr_vs_k', 'keel'],
       ['load_led7digit', 500, 7, 7, 10, 'led7digit', 'keel'],
       ['load_movement_libras', 360, 90, 90, 15, 'movement_libras', 'keel'],
       ['load_newthyroid', 215, 5, 5, 3, 'newthyroid', 'keel'],
       ['load_nursery', 12960, 26, 8, 5, 'nursery', 'keel'],
       ['load_page_blocks', 5472, 10, 10, 5, 'page_blocks', 'keel'],
       ['load_post_operative', 87, 21, 8, 3, 'post_operative', 'keel'],
       ['load_segment', 2310, 18, 19, 7, 'segment', 'keel'],
       ['load_splice', 3190, 287, 60, 3, 'splice', 'keel'],
       ['load_tae', 151, 5, 5, 3, 'tae', 'keel'],
       ['load_vowel', 990, 13, 13, 11, 'vowel', 'keel'],
       ['load_zoo', 101, 21, 16, 7, 'zoo', 'keel']], dtype=object)

summary_columns=['loader_function', 'n', 'n_attr_encoded', 'n_attr_raw', 'n_classes',
       'name', 'reference_key']