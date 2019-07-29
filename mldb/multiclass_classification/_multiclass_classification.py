import numpy as np
import pandas as pd

from mldb._io import read_csv_data, read_arff_data, construct_return_set, citations

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
          'get_filtered_data_loaders',
          'get_data_loaders']

def load_glass(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "glass", return_X_y, encode, citation= citations['krnn'], name= "glass", verbose= verbose, problem_type='multiclass')

def load_satimage(return_X_y= False, encode= True, verbose= False):
    db0= read_csv_data('data/classification/satimage/sat.trn.txt', sep= ' ')
    db1= read_csv_data('data/classification/satimage/sat.tst.txt', sep= ' ')
    db= pd.concat([db0, db1])
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "SATIMAGE", return_X_y, encode, citation= citations['krnn'], name= "SATIMAGE", verbose= verbose, problem_type='multiclass')

def load_ecoli(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/ecoli/ecoli.data.txt', delim_whitespace=True)
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    print(db.columns)
    return construct_return_set(db, "ecoli", return_X_y, encode, citation= citations['krnn'], name= "ecoli", verbose= verbose, problem_type='multiclass')

def load_abalone(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/abalone/abalone.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "abalone", return_X_y, encode, citation= citations['krnn'], name= "abalone", verbose= verbose, problem_type='multiclass')

def load_yeast(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/yeast/yeast.data.txt', delim_whitespace=True)
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "yeast", return_X_y, encode, citation= citations['krnn'], name= "yeast", verbose= verbose, problem_type='multiclass')

def load_winequality_red(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "winequality_red", return_X_y, encode, citation= citations['krnn'], name= "winequality_red", verbose= verbose, problem_type='multiclass')

def load_winequality_white(return_X_y= False, encode= True, verbose= False):
    db= read_csv_data('data/classification/glass/glass.data.txt')
    db.columns= list(db.columns[:-1]) + ['target']
    del db[db.columns[0]]
    
    return construct_return_set(db, "winequality_white", return_X_y, encode, citation= citations['krnn'], name= "winequality_white", verbose= verbose, problem_type='multiclass')

def load_automobile(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/automobile/automobile.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "automobile", return_X_y, encode, citation= citations['keel'], name= "automobile", verbose= verbose, problem_type='multiclass')

def load_balance(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/balance/balance.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "balance", return_X_y, encode, citation= citations['keel'], name= "balance", verbose= verbose, problem_type='multiclass')

def load_car(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/car/car.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "car", return_X_y, encode, citation= citations['keel'], name= "car", verbose= verbose, problem_type='multiclass')

def load_cleveland(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/cleveland/cleveland.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "cleveland", return_X_y, encode, citation= citations['keel'], name= "cleveland", verbose= verbose, problem_type='multiclass')

def load_contraceptive(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/contraceptive/contraceptive.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "contraceptive", return_X_y, encode, citation= citations['keel'], name= "contraceptive", verbose= verbose, problem_type='multiclass')

def load_dermatology(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/dermatology/dermatology.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "dermatology", return_X_y, encode, citation= citations['keel'], name= "dermatology", verbose= verbose, problem_type='multiclass')

def load_fars(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/fars/fars.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "fars", return_X_y, encode, citation= citations['keel'], name= "fars", verbose= verbose, problem_type='multiclass')

def load_flare(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/flare/flare.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "flare", return_X_y, encode, citation= citations['keel'], name= "flare", verbose= verbose, problem_type='multiclass')

def load_hayes_roth(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/hayes-roth/hayes-roth.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "hayes_roth", return_X_y, encode, citation= citations['keel'], name= "hayes_roth", verbose= verbose, problem_type='multiclass')

def load_kr_vs_k(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/kr-vs-k/kr-vs-k.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "kr_vs_k", return_X_y, encode, citation= citations['keel'], name= "kr_vs_k", verbose= verbose, problem_type='multiclass')

def load_led7digit(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/led7digit/led7digit.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "led7digit", return_X_y, encode, citation= citations['keel'], name= "led7digit", verbose= verbose, problem_type='multiclass')

def load_movement_libras(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/movement_libras/movement_libras.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "movement_libras", return_X_y, encode, citation= citations['keel'], name= "movement_libras", verbose= verbose, problem_type='multiclass')

def load_newthyroid(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/newthyroid/newthyroid.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "newthyroid", return_X_y, encode, citation= citations['keel'], name= "newthyroid", verbose= verbose, problem_type='multiclass')

def load_nursery(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/nursery/nursery.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "nursery", return_X_y, encode, citation= citations['keel'], name= "nursery", verbose= verbose, problem_type='multiclass')

def load_page_blocks(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/page-blocks/page-blocks.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "page_blocks", return_X_y, encode, citation= citations['keel'], name= "page_blocks", verbose= verbose, problem_type='multiclass')

def load_post_operative(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/post-operative/post-operative.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "post_operative", return_X_y, encode, citation= citations['keel'], name= "post_operative", verbose= verbose, problem_type='multiclass')

def load_segment(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/segment/segment.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "segment", return_X_y, encode, citation= citations['keel'], name= "segment", verbose= verbose, problem_type='multiclass')

def load_splice(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/splice/splice.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "splice", return_X_y, encode, citation= citations['keel'], name= "splice", verbose= verbose, problem_type='multiclass')

def load_tae(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/tae/tae.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "tae", return_X_y, encode, citation= citations['keel'], name= "tae", verbose= verbose, problem_type='multiclass')

def load_vowel(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/vowel/vowel.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "vowel", return_X_y, encode, citation= citations['keel'], name= "vowel", verbose= verbose, problem_type='multiclass')

def load_zoo(return_X_y= False, encode= True, verbose= False):
    data, meta= read_arff_data('data/classification/zoo/zoo.dat')
    db= pd.DataFrame(data)
    db.columns= list(db.columns[:-1]) + ['target']
    
    return construct_return_set(db, "zoo", return_X_y, encode, citation= citations['keel'], name= "zoo", verbose= verbose, problem_type='multiclass')

def summary(include_citation= True, ratio_dist= False, subset= 'study', verbose= False):
    results= []
    # fixing the globals dictionary keys
    d= __all__
    
    for func_name in d:
        if func_name.startswith('load_') and not func_name.startswith('load_artificial'):
            data_not_encoded= globals()[func_name](return_X_y= False, encode= False, verbose= verbose)
            data_encoded= globals()[func_name](return_X_y= False, encode= True, verbose= verbose)
            
            labels, counts= np.unique(data_encoded['target'], return_counts= True)
            
            result= {'loader_function': globals()[func_name],
                            'name': data_not_encoded['name'],
                            'len': len(data_not_encoded['data']),
                            'non_encoded_n_attr': len(data_not_encoded['data'][0]),
                            'encoded_n_attr': len(data_encoded['data'][0]),
                            'n_classes': len(labels),
                            'n_majority': np.max(counts),
                            'n_minority': np.min(counts),
                            'imbalance_ratio': np.max(counts)/np.min(counts)}
            
            if ratio_dist:
                from sklearn.neighbors import NearestNeighbors
                from sklearn.preprocessing import StandardScaler
                nn= NearestNeighbors(n_neighbors= 2)
                X= data_encoded['data']
                
                X_min= X[data_encoded['target'] == labels[counts == np.min(counts)][0]]
                X_maj= X[data_encoded['target'] == labels[counts == np.max(counts)][0]]
                
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
    descriptors= summary()
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
