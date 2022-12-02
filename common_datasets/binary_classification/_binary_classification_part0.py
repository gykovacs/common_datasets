"""
This module contains the binary classification data loaders
"""

from .._io import (load_arff_template_binary)

__all__= ['load_abalone_17_vs_7_8_9_10',
            'load_abalone_19_vs_10_11_12_13',
            'load_abalone_20_vs_8_9_10',
            'load_abalone_21_vs_8',
            'load_abalone_3_vs_11',
            'load_abalone19',
            'load_abalone9_18',
            'load_ecoli_0_1_3_7_vs_2_6',
            'load_ecoli_0_1_4_6_vs_5',
            'load_ecoli_0_1_4_7_vs_2_3_5_6',
            'load_ecoli_0_1_4_7_vs_5_6',
            'load_ecoli_0_1_vs_2_3_5',
            'load_ecoli_0_1_vs_5',
            'load_ecoli_0_2_3_4_vs_5',
            'load_ecoli_0_2_6_7_vs_3_5',
            'load_ecoli_0_3_4_6_vs_5',
            'load_ecoli_0_3_4_7_vs_5_6',
            'load_ecoli_0_3_4_vs_5',
            'load_ecoli_0_4_6_vs_5',
            'load_ecoli_0_6_7_vs_3_5',
            'load_ecoli_0_6_7_vs_5',
            'load_ecoli4',
            'load_ecoli_0_vs_1',
            'load_ecoli1',
            'load_ecoli2',
            'load_ecoli3',
            'load_glass_0_1_4_6_vs_2',
            'load_glass_0_1_5_vs_2',
            'load_glass_0_1_6_vs_2',
            'load_glass_0_1_6_vs_5',
            'load_glass_0_4_vs_5',
            'load_glass_0_6_vs_5',
            'load_glass2',
            'load_glass4',
            'load_glass5',
            'load_glass_0_1_2_3_vs_4_5_6',
            'load_glass0',
            'load_glass1',
            'load_glass6',
            'load_yeast_0_2_5_6_vs_3_7_8_9',
            'load_yeast_0_2_5_7_9_vs_3_6_8',
            'load_yeast_0_3_5_9_vs_7_8',
            'load_yeast_0_5_6_7_9_vs_4',
            'load_yeast_1_2_8_9_vs_7',
            'load_yeast_1_4_5_8_vs_7',
            'load_yeast_1_vs_7',
            'load_yeast_2_vs_4',
            'load_yeast_2_vs_8',
            'load_yeast4',
            'load_yeast5',
            'load_yeast6',
            'load_yeast1',
            'load_yeast3',
            'load_winequality_red_3_vs_5',
            'load_winequality_red_4',
            'load_winequality_red_8_vs_6',
            'load_winequality_red_8_vs_6_7',
            'load_winequality_white_3_9_vs_5',
            'load_winequality_white_3_vs_7',
            'load_winequality_white_9_vs_4']

def load_abalone_17_vs_7_8_9_10():
    """
    Loads the abalone_17_vs_7_8_9_10 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/abalone-17_vs_7-8-9-10/abalone-17_vs_7-8-9-10.dat'
    return load_arff_template_binary(path=path,
                                        name='abalone-17_vs_7-8-9-10',
                                        target_label='Class')

def load_abalone_19_vs_10_11_12_13():
    """
    Loads the abalone_19_vs_10_11_12_13 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/abalone-19_vs_10-11-12-13/abalone-19_vs_10-11-12-13.dat'
    return load_arff_template_binary(path=path,
                                        name='abalone-19_vs_10-11-12-13',
                                        target_label='Class')

def load_abalone_20_vs_8_9_10():
    """
    Loads the abalone_20_vs_8_9_10 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/abalone-20_vs_8-9-10/abalone-20_vs_8-9-10.dat'
    return load_arff_template_binary(path=path,
                                        name='abalone-20_vs_8_9_10',
                                        target_label='Class')

def load_abalone_21_vs_8():
    """
    Loads the abalone_21_vs_8 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/abalone-21_vs_8/abalone-21_vs_8.dat'
    return load_arff_template_binary(path=path,
                                        name='abalone-22_vs_8',
                                        target_label='Class')

def load_abalone_3_vs_11():
    """
    Loads the abalone_3_vs_11 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/abalone-3_vs_11/abalone-3_vs_11.dat'
    return load_arff_template_binary(path=path,
                                        name='abalone-3_vs_11',
                                        target_label='Class')

def load_abalone19():
    """
    Loads the abalone19 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/abalone19/abalone19.dat'
    return load_arff_template_binary(path=path,
                                        name='abalone19',
                                        target_label='Class')

def load_abalone9_18():
    """
    Loads the abalone9_18 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/abalone9-18/abalone9-18.dat'
    return load_arff_template_binary(path=path,
                                        name='abalone9_18',
                                        target_label='Class')

def load_ecoli_0_1_3_7_vs_2_6():
    """
    Loads the ecoli_0_1_3_7_vs_2_6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-1-3-7_vs_2-6/ecoli-0-1-3-7_vs_2-6.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_1_3_7_vs_2_6',
                                        target_label='Class')

def load_ecoli_0_1_4_6_vs_5():
    """
    Loads the ecoli_0_1_4_6_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-1-4-6_vs_5/ecoli-0-1-4-6_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_1_4_6_vs_5',
                                        target_label='class')

def load_ecoli_0_1_4_7_vs_2_3_5_6():
    """
    Loads the ecoli_0_1_4_7_vs_2_3_5_6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-1-4-7_vs_2-3-5-6/ecoli-0-1-4-7_vs_2-3-5-6.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_1_4_7_vs_2_3_5_6',
                                        target_label='class')

def load_ecoli_0_1_4_7_vs_5_6():
    """
    Loads the ecoli_0_1_4_7_vs_5_6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-1-4-7_vs_5-6/ecoli-0-1-4-7_vs_5-6.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_1_4_7_vs_5_6',
                                        target_label='class')

def load_ecoli_0_1_vs_2_3_5():
    """
    Loads the ecoli_0_1_vs_2_3_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-1_vs_2-3-5/ecoli-0-1_vs_2-3-5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_1_vs_2_3_5',
                                        target_label='class')

def load_ecoli_0_1_vs_5():
    """
    Loads the ecoli_0_1_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-1_vs_5/ecoli-0-1_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_1_vs_5',
                                        target_label='class')

def load_ecoli_0_2_3_4_vs_5():
    """
    Loads the ecoli_0_2_3_4_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-2-3-4_vs_5/ecoli-0-2-3-4_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_2_3_4_vs_5',
                                        target_label='class')

def load_ecoli_0_2_6_7_vs_3_5():
    """
    Loads the ecoli_0_2_6_7_vs_3_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-2-6-7_vs_3-5/ecoli-0-2-6-7_vs_3-5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_2_6_7_vs_3_5',
                                        target_label='class')

def load_ecoli_0_3_4_6_vs_5():
    """
    Loads the ecoli_0_3_4_6_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-3-4-6_vs_5/ecoli-0-3-4-6_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_3_4_6_vs_5',
                                        target_label='class')

def load_ecoli_0_3_4_7_vs_5_6():
    """
    Loads the ecoli_0_3_4_7_vs_5_6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-3-4-7_vs_5-6/ecoli-0-3-4-7_vs_5-6.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_3_4_7_vs_5_6',
                                        target_label='class')

def load_ecoli_0_3_4_vs_5():
    """
    Loads the ecoli_0_3_4_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-3-4_vs_5/ecoli-0-3-4_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_3_4_vs_5',
                                        target_label='class')

def load_ecoli_0_4_6_vs_5():
    """
    Loads the ecoli_0_4_6_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-4-6_vs_5/ecoli-0-4-6_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_4_6_vs_5',
                                        target_label='class')

def load_ecoli_0_6_7_vs_3_5():
    """
    Loads the ecoli_0_6_7_vs_3_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-6-7_vs_3-5/ecoli-0-6-7_vs_3-5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_6_7_vs_3_5',
                                        target_label='class')

def load_ecoli_0_6_7_vs_5():
    """
    Loads the ecoli_0_6_7_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0-6-7_vs_5/ecoli-0-6-7_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_6_7_vs_5',
                                        target_label='class')

def load_ecoli4():
    """
    Loads the ecoli4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli4/ecoli4.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli4',
                                        target_label='Class')

def load_ecoli_0_vs_1():
    """
    Loads the ecoli_0_vs_1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli-0_vs_1/ecoli-0_vs_1.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli_0_vs_1',
                                        target_label="Class")

def load_ecoli1():
    """
    Loads the ecoli1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli1/ecoli1.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli1',
                                        target_label='Class')

def load_ecoli2():
    """
    Loads the ecoli2 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli2/ecoli2.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli2',
                                        target_label='Class')

def load_ecoli3():
    """
    Loads the ecoli3 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/ecoli3/ecoli3.dat'
    return load_arff_template_binary(path=path,
                                        name='ecoli3',
                                        target_label='Class')

def load_glass_0_1_4_6_vs_2():
    """
    Loads the glass_0_1_4_6_vs_2 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass-0-1-4-6_vs_2/glass-0-1-4-6_vs_2.dat'
    return load_arff_template_binary(path=path,
                                        name='glass_0_1_4_6_vs_2',
                                        target_label='typeGlass')

def load_glass_0_1_5_vs_2():
    """
    Loads the glass_0_1_5_vs_2 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass-0-1-5_vs_2/glass-0-1-5_vs_2.dat'
    return load_arff_template_binary(path=path,
                                        name='glass_0_1_5_vs_2',
                                        target_label='typeGlass')

def load_glass_0_1_6_vs_2():
    """
    Loads the glass_0_1_6_vs_2 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass-0-1-6_vs_2/glass-0-1-6_vs_2.dat'
    return load_arff_template_binary(path=path,
                                        name='glass_0_1_6_vs_2',
                                        target_label='Class')

def load_glass_0_1_6_vs_5():
    """
    Loads the glass_0_1_6_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass-0-1-6_vs_5/glass-0-1-6_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='glass_0_1_6_vs_5',
                                        target_label='Class')

def load_glass_0_4_vs_5():
    """
    Loads the glass_0_4_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass-0-4_vs_5/glass-0-4_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='glass_0_4_vs_5',
                                        target_label='typeGlass')

def load_glass_0_6_vs_5():
    """
    Loads the glass_0_6_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass-0-6_vs_5/glass-0-6_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name='glass_0_6_vs_5',
                                        target_label='typeGlass')

def load_glass2():
    """
    Loads the glass2 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass2/glass2.dat'
    return load_arff_template_binary(path=path,
                                        name='glass2',
                                        target_label='Class')

def load_glass4():
    """
    Loads the glass4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass4/glass4.dat'
    return load_arff_template_binary(path=path,
                                        name='glass4',
                                        target_label='Class')

def load_glass5():
    """
    Loads the glass5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass5/glass5.dat'
    return load_arff_template_binary(path=path,
                                        name='glass5',
                                        target_label='Class')

def load_glass_0_1_2_3_vs_4_5_6():
    """
    Loads the glass_0_1_2_3_vs_4_5_6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass-0-1-2-3_vs_4-5-6/glass-0-1-2-3_vs_4-5-6.dat'
    return load_arff_template_binary(path=path,
                                        name='glass_0_1_2_3_vs_4_5_6',
                                        target_label='Class')

def load_glass0():
    """
    Loads the glass0 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass0/glass0.dat'
    return load_arff_template_binary(path=path,
                                        name='glass0',
                                        target_label='Class')

def load_glass1():
    """
    Loads the glass1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass1/glass1.dat'
    return load_arff_template_binary(path=path,
                                        name='glass1',
                                        target_label='Class')

def load_glass6():
    """
    Loads the glass6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/glass6/glass6.dat'
    return load_arff_template_binary(path=path,
                                        name='glass6',
                                        target_label='Class')

def load_yeast_0_2_5_6_vs_3_7_8_9():
    """
    Load the yeast-0-2-5-6_vs_3-7-8-9 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-0-2-5-6_vs_3-7-8-9/yeast-0-2-5-6_vs_3-7-8-9.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-0-2-5-6_vs_3-7-8-9",
                                        target_label='class')

def load_yeast_0_2_5_7_9_vs_3_6_8():
    """
    Load the yeast-0-2-5-7-9_vs_3-6-8 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-0-2-5-7-9_vs_3-6-8/yeast-0-2-5-7-9_vs_3-6-8.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-0-2-5-7-9_vs_3-6-8",
                                        target_label='class')

def load_yeast_0_3_5_9_vs_7_8():
    """
    Load the yeast-0-3-5-9_vs_7-8 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-0-3-5-9_vs_7-8/yeast-0-3-5-9_vs_7-8.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-0-3-5-9_vs_7-8",
                                        target_label='class')

def load_yeast_0_5_6_7_9_vs_4():
    """
    Load the yeast-0-5-6-7-9_vs_4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-0-5-6-7-9_vs_4/yeast-0-5-6-7-9_vs_4.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-0-5-6-7-9_vs_4",
                                        target_label='Class')

def load_yeast_1_2_8_9_vs_7():
    """
    Load the yeast-1-2-8-9_vs_7 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-1-2-8-9_vs_7/yeast-1-2-8-9_vs_7.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-1-2-8-9_vs_7",
                                        target_label='Class')

def load_yeast_1_4_5_8_vs_7():
    """
    Load the yeast-1-2-8-9_vs_7 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-1-4-5-8_vs_7/yeast-1-4-5-8_vs_7.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-1-4-5-8_vs_7",
                                        target_label='Class')

def load_yeast_1_vs_7():
    """
    Load the yeast-1_vs_7 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-1_vs_7/yeast-1_vs_7.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-1_vs_7",
                                        target_label='Class')

def load_yeast_2_vs_4():
    """
    Load the yeast-2_vs_4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-2_vs_4/yeast-2_vs_4.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-2_vs_4",
                                        target_label='Class')

def load_yeast_2_vs_8():
    """
    Load the yeast-2_vs_8 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast-2_vs_8/yeast-2_vs_8.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast-2_vs_8",
                                        target_label='Class')

def load_yeast4():
    """
    Load the yeast4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast4/yeast4.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast4",
                                        target_label='Class')

def load_yeast5():
    """
    Load the yeast5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast5/yeast5.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast5",
                                        target_label='Class')

def load_yeast6():
    """
    Load the yeast6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast6/yeast6.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast6",
                                        target_label='Class')

def load_yeast1():
    """
    Load the yeast1 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast1/yeast1.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast1",
                                        target_label='Class')

def load_yeast3():
    """
    Load the yeast3 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/yeast3/yeast3.dat'
    return load_arff_template_binary(path=path,
                                        name="yeast3",
                                        target_label='Class')

def load_winequality_red_3_vs_5():
    """
    Load the winequality_red_3_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/winequality-red-3_vs_5/winequality-red-3_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name="winequality-red-3_vs_5",
                                        target_label='Class')

def load_winequality_red_4():
    """
    Load the winequality-red-4 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/winequality-red-4/winequality-red-4.dat'
    return load_arff_template_binary(path=path,
                                        name="winequality-red-4",
                                        target_label='Class')

def load_winequality_red_8_vs_6():
    """
    Load the winequality-red-8_vs_6 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/winequality-red-8_vs_6/winequality-red-8_vs_6.dat'
    return load_arff_template_binary(path=path,
                                        name="winequality-red-8_vs_6",
                                        target_label='Class')

def load_winequality_red_8_vs_6_7():
    """
    Load the winequality-red-8_vs_6-7 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/winequality-red-8_vs_6-7/winequality-red-8_vs_6-7.dat'
    return load_arff_template_binary(path=path,
                                        name="winequality-red-8_vs_6-7",
                                        target_label='Class')

def load_winequality_white_3_9_vs_5():
    """
    Load the winequality-white-3-9_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/winequality-white-3-9_vs_5/winequality-white-3-9_vs_5.dat'
    return load_arff_template_binary(path=path,
                                        name="winequality-white-3-9_vs_5",
                                        target_label='Class')

def load_winequality_white_3_vs_7():
    """
    Load the winequality-white-3-9_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/winequality-white-3_vs_7/winequality-white-3_vs_7.dat'
    return load_arff_template_binary(path=path,
                                        name="winequality-white-3_vs_7",
                                        target_label='Class')

def load_winequality_white_9_vs_4():
    """
    Load the winequality-white-3-9_vs_5 dataset

    Returns:
        dict: the dataset in sklearn.datasets representation
    """
    path = 'data/classification/winequality-white-9_vs_4/winequality-white-9_vs_4.dat'
    return load_arff_template_binary(path=path,
                                        name="winequality-white-9_vs_4",
                                        target_label='Class')
