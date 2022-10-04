===========
ml_processor
===========

.. list-table::

	* - .. figure:: images/output_24_0.png

.. list-table::

	* - .. figure:: images/output_56_0.png

**ml_processor** is a library written in python for perfoming most of the common data preprocessing tasks involved in building machine learning models. It includes methods for:

* perfroming necessary data transformation desired for machine learning modesl
* hyperparameter tunning using different methods notably One Hot encoding and WOE transformation
* automted machine learning model fitting and model performacne evaluation

Installation
============

To install the latest release of ml_processor from PyPi:

.. code-block:: test
	
	pip install ml_processor

Depencies
---------

ml-processor requires

* pandas
* numpy
* matplotlib
* seaborn
* logging
* json
* dotenv
* sklearn
* optbinning
* pickle
* joblib
* snowflake
* sqlalchemy
* xgboost
* statsmodels
* hyperopt
* scipy

Getting started
===============

Tutorials
---------

Example: config
---------------

**config** from the configuration sub-module provides a conveneient way for working with information such as credentials that one might want to keep secret and not include into their script. It also provides an easy of logging information both to the console and creating of log files.


get_credentials
_______________

Takes as an argument a path to the location of a stored .env file and returns the contents in the file.

.. code-block:: python

   from ml_processor.configuration import config

   >>> config.get_credentials('./examples/.env')

   OrderedDict([('username', 'email.example.com'), ('password', 'myPassword')])

Example: eda_data_quality
-------------------------

Checks dataset aganist specific rules and assigns a data quality score. 

Let us load the `Home Credit Default Risk <https://www.kaggle.com/competitions/home-credit-default-risk/data?select=application_train.csv>`_ dataset provided on kaggle and perform qaulity checks on it

.. code-block:: python
   
   import pandas as pd

   df = pd.read_csv('./data/application_train.csv')

   >>> eda_data_quality(df).head()

.. code-block:: text
  
               name      dtype   status  selected n_bins        iv        js      gini quality_score
   0   ext_source_3  numerical  OPTIMAL      True      6  0.317153  0.038595  0.307837      0.927042
   1   ext_source_2  numerical  OPTIMAL      True      8  0.307775  0.037361  0.306041       0.91906
   2   ext_source_1  numerical  OPTIMAL      True      3  0.127146  0.015586  0.169518      0.387648
   3  days_employed  numerical  OPTIMAL      True      7   0.11093  0.013768  0.184596      0.399716
   4     days_birth  numerical  OPTIMAL      True      7  0.085185  0.010572   0.16357       0.32957

