from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='mldb',
      version='0.1.1',
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
      license='MIT',
      packages=['mldb',
                'mldb.regression',
                'mldb.clustering',
                'mldb.binary_classification',
                'mldb.multiclass_classification'],
      install_requires=[
              'numpy',
              'pandas',
              'scipy',
              'sklearn'
              ],
      py_modules=['mldb',
		    'mldb.regression',
		    'mldb.clustering',
		    'mldb.binary_classification',
		    'mldb.multiclass_classification'],
      zip_safe=False,
      package_dir= {'mldb': 'mldb',
		    'mldb.regression': 'mldb/regression',
		    'mldb.clustering': 'mldb/clustering',
		    'mldb.binary_classification': 'mldb/binary_classification',
		    'mldb.multiclass_classification': 'mldb/multiclass_classification'},
      package_data= {'mldb': ['mldb/data/*/*']}
      )
