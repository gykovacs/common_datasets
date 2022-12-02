import os
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

version_file= os.path.join('common_datasets', '_version.py')
__version__= "0.0.0"
with open(version_file) as f:
    exec(f.read())

setup(name='common_datasets',
      version=__version__,
      description='common_datasets',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Alpha',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      url='http://github.com/gykovacs/common_datasets',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      packages=['common_datasets',
                'common_datasets.regression',
                'common_datasets.clustering',
                'common_datasets.binary_classification',
                'common_datasets.multiclass_classification'],
      install_requires=[
              'numpy',
              'pandas',
              'scipy',
              'scikit-learn',
              'openpyxl'
              ],
      py_modules=['common_datasets',
		    'common_datasets.regression',
		    'common_datasets.clustering',
		    'common_datasets.binary_classification',
		    'common_datasets.multiclass_classification'],
      zip_safe=False,
      package_dir= {'common_datasets': 'common_datasets',
		    'common_datasets.regression': 'common_datasets/regression',
		    'common_datasets.clustering': 'common_datasets/clustering',
		    'common_datasets.binary_classification': 'common_datasets/binary_classification',
		    'common_datasets.multiclass_classification': 'common_datasets/multiclass_classification'},
      package_data= {'common_datasets': ['data/*/*/*', 'common_datasets/data/*', 'common_datasets/data/data_level.txt']},
      include_package_data=True
      )
