import os
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

version_file= os.path.join('mldb', '__version__.py')
__version__= "0.0.0"
with open(version_file) as f:
    exec(f.read())

setup(name='mldb',
      version=__version__,
      description='mldb',
      long_description=readme(),
      classifiers=[
              'Development Status :: 3 - Alpha',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python',
              'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      url='http://github.com/gykovacs/mldb',
      author='Gyorgy Kovacs',
      author_email='gyuriofkovacs@gmail.com',
      packages=['mldb',
                'mldb.regression',
                'mldb.clustering',
                'mldb.binary_classification',
                'mldb.multiclass_classification'],
      install_requires=[
              'numpy',
              'pandas',
              'scipy',
              'sklearn',
              'openpyxl'
              ],
      #py_modules=['mldb',
	#	    'mldb.regression',
	#	    'mldb.clustering',
	#	    'mldb.binary_classification',
	#	    'mldb.multiclass_classification'],
      zip_safe=False,
      package_dir= {'mldb': 'mldb',
		    'mldb.regression': 'mldb/regression',
		    'mldb.clustering': 'mldb/clustering',
		    'mldb.binary_classification': 'mldb/binary_classification',
		    'mldb.multiclass_classification': 'mldb/multiclass_classification'},
      package_data= {'mldb': ['mldb/data/*/*']}
      )
