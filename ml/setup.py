
from setuptools import setup
import os



path = os.path.join(os.environ.get('HOME'), 'Library/CloudStorage/OneDrive-MoneseLtd/GW/libraries/ml/README.md')

with open(path, "r") as fh:
    long_description = fh.read()

setup(
    # name that will be imported, can be different from code file name
    name='ml_processor',

    version='0.0.6',

    description='Includes functions for performing econometrics tasks',

    # code file name without file extension
    py_modules=['configuration', 'eda_analysis', 'encoders', 'jsonSerializer', 'model_training' 'outliers' 'snowflake_processor'],

    # directory in which code file is stored
    package_dir={'':'src'},

    long_description=long_description,

    long_description_content_type="text/markdown",

    author="Geofrey Wanyama",

    author_email="wanyamag17@gmail.com",

    url="https://github.com/G-Geofrey/ml_processor.git",

    zip_safe=False,

    classifers=[ 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
        ],

    #  install_requires=[
    #     'pandas>=1.3.4',
    #      'numpy>=1.20.3',
    #      'matplotlib>=3.4.3',
    #      'seaborn>=0.11.2',
    #      'statsmodels>=0.12.2',
    #      'stargazer'
    # ]


    )