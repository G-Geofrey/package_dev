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
    
   2022-10-03 23:15:19,318:INFO:rule_1 : More than 50% of the data missing
   2022-10-03 23:15:19,319:INFO:rule_2 : Missing some data
   2022-10-03 23:15:19,319:INFO:rule_3 : 75% of the data is the same and equal to the minimum
   2022-10-03 23:15:19,319:INFO:rule_4 : 50% of the data is the same and equal to the minimum
   2022-10-03 23:15:19,320:INFO:rule_5 : Has negative values
   2022-10-03 23:15:19,320:INFO:rule_6 : Possible wrong data type
  
                                type  unique  missing pct.missing      mean  min  25%  50%     75%  max  rule_1  rule_2  rule_3  rule_4  rule_5  rule_6  quality_score
   elevators_mode            float64      26   163891       53.3%  0.074490  0.0  0.0  0.0  0.1208  1.0       1       1       0       1       0       1       0.400000
   nonlivingapartments_avg   float64     386   213514       69.4%  0.008809  0.0  0.0  0.0  0.0039  1.0       1       1       0       1       0       0       0.528571
   elevators_avg             float64     257   163891       53.3%  0.078942  0.0  0.0  0.0  0.1200  1.0       1       1       0       1       0       0       0.528571
   nonlivingapartments_mode  float64     167   213514       69.4%  0.008076  0.0  0.0  0.0  0.0039  1.0       1       1       0       1       0       0       0.528571
   elevators_medi            float64      46   163891       53.3%  0.078078  0.0  0.0  0.0  0.1200  1.0       1       1       0       1       0       0       0.528571

