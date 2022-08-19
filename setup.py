import os
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

version_file= os.path.join('mldatasets', '__version__.py')
__version__= "0.0.0"
with open(version_file) as f:
    exec(f.read())

setup(name='mldatasets',
      version=__version__,
      description='mldatasets',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Alpha',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      url='http://github.com/gykovacs/mldatasets',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      packages=['mldatasets',
                'mldatasets.regression',
                'mldatasets.clustering',
                'mldatasets.binary_classification',
                'mldatasets.multiclass_classification'],
      install_requires=[
              'numpy',
              'pandas',
              'scipy',
              'sklearn',
              'openpyxl'
              ],
      #py_modules=['mldatasets',
	#	    'mldatasets.regression',
	#	    'mldatasets.clustering',
	#	    'mldatasets.binary_classification',
	#	    'mldatasets.multiclass_classification'],
      zip_safe=False,
      package_dir= {'mldatasets': 'mldatasets',
		    'mldatasets.regression': 'mldatasets/regression',
		    'mldatasets.clustering': 'mldatasets/clustering',
		    'mldatasets.binary_classification': 'mldatasets/binary_classification',
		    'mldatasets.multiclass_classification': 'mldatasets/multiclass_classification'},
      package_data= {'mldatasets': ['mldatasets/data/*/*']}
      )
