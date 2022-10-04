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
  
	   			Bin		Count	Count (%)	Non-event	Event	Event rate	WoE	IV	JS
   0	(-inf, 0.34)	34244	0.111359	29618	4626	0.135089	-0.575796	0.046968	0.005791
   1	[0.34, 0.57)	45066	0.146551	41788	3278	0.072738	0.11289	0.001781	0.000223
   2	[0.57, inf)		54823	0.17828	52673	2150	0.039217	0.76615	0.076404	0.009324
   3	Special			0	0	0	0	0	0	0	0
   4	Missing			173378	0.563811	158607	14771	0.085195	-0.058722	0.001993	0.000249
   Totals				307511	1	282686	24825	0.080729		0.127146	0.015586

