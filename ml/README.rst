===========
ml_processor
===========

.. list-table::

	* - .. figure:: images/output_38_0.png

.. list-table::

	* - .. figure:: images/output_54_0.png

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

	022-09-30 20:17:15,974:INFO:rule_1 : More than 50% of the data missing
	2022-09-30 20:17:15,975:INFO:rule_2 : Missing some data
	2022-09-30 20:17:15,975:INFO:rule_3 : 75% of the data is the same and equal to the minimum
	2022-09-30 20:17:15,976:INFO:rule_4 : 50% of the data is the same and equal to the minimum
	2022-09-30 20:17:15,976:INFO:rule_5 : Has negative values
	2022-09-30 20:17:15,976:INFO:rule_6 : Possible wrong data type

	type	unique	missing	pct.missing	mean	min	25%	50%	75%	max	rule_1	rule_2	rule_3	rule_4	rule_5	rule_6	quality_score
	reg_city_not_live_city	int64	2	0	0.0%	0.0372	0.0000	0.0000	0.0000	0.0000	1.0000	0	0	1	0	0	1	0.6429
	amt_req_credit_bureau_qrt	float64	8	0	0.0%	0.2581	0.0000	0.0000	0.0000	0.0000	8.0000	0	0	1	0	0	1	0.6429
	flag_document_10	int64	2	0	0.0%	0.0001	0.0000	0.0000	0.0000	0.0000	1.0000	0	0	1	0	0	1	0.6429
	flag_mobil	int64	1	0	0.0%	1.0000	1.0000	1.0000	1.0000	1.0000	1.0000	0	0	1	0	0	1	0.6429
	flag_document_4	int64	1	0	0.0%	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000	0	0	1	0	0	1	0.6429



