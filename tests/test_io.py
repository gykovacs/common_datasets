"""
Testing the IO and encoding functionalities
"""

import pandas as pd

from common_datasets._io import coalesce, DataPreprocessor

def test_coalesce():
    """
    Testing the coalesce function
    """

    assert coalesce(1, 2) == 1

    assert coalesce(None, 2) == 2

def test_log_grid():
    """
    Testing the log grid directive
    """
    
    dp = DataPreprocessor(pd.DataFrame({'a': [0, 0, 1, 0, 2, 0, 0, 0, 0, 1],
                                        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]}),
                          grid_threshold='log2',
                          target_label='target',
                          name='test_dataset',
                          feature_types={'a': 'numeric',
                                         'target': 'numeric'})
    
    dataset = dp.get_dataset()
    
    assert dataset['grid'][0]