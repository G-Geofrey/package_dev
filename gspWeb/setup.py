
from setuptools import setup, find_packages
import os


path = os.path.join(os.environ.get('HOME'), 'Desktop/wg/projects/package_dev/gspWeb/README.rst')

with open(path, "r") as fh:
    long_description = fh.read()

setup(
    name='gspWeb',

    version='0.0.6',

    description='Fitting ols model in python',

    packages=['gspWeb', 'gspWeb.econometrics'],

    # directory in which code file is stored
    package_dir={'':'src'},

    long_description=long_description,

    long_description_content_type="text/x-rst",

    author="Geofrey Wanyama",

    author_email="wanyamag17@gmail.com",

    url="",

    zip_safe=False,

    classifers=[ 
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
        ],

     install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "plotly==5.14.0",
        "scipy",
        "seaborn",
        "statsmodels==0.13.5",
    ]
)