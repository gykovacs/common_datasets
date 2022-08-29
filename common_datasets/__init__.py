"""
The root module of the package
"""

from ._io import references

from . import binary_classification
from . import multiclass_classification
from . import regression
from . import clustering

from __version__ import *
