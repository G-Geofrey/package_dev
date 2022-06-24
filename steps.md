
# Developing pyhton package

## write python script with package code e.g package_name.py
```python
def say_hello(name):
    print(f"hello {name}"
```
## save the file in a source directory named  "src" directory
```
src
├──  package_file_name.py
 
```
## create git ignore file and save it in the same directory that has "src" directory
	- create git ignore file
	- open gitignore.io in the browser
	- type python and create
	- copy file and paste in git ignore file

## create LICENSE.txt file
	https://choosealicense.com

## write a README.md
	- title of project
	- short description
	- how to install

## create setup.py file and save it in the same directory that has the source directory
```python

from setuptools import setup

setup(
    # name that will be imported, can be different from code file name
    name='py_econometrics',

    version='0.0.01',

    description='Includes functions for performing econometrics tasks',
se
    # code file name without file extension
    py_modules=['econometrics'],

    # directory in which code file is stored
    package_dir={'':'src'}
    )
```

## add README.md to `setup.py` file
```python

from setuptools import setup

with open("/Users/geofrey.wanyama/Desktop/libraries/econometrics/README.md", "r") as fh:
    long_description = fh.read()

setup(

	...,

    # directory in which code file is stored
    package_dir={'':'src'},

    long_description=long_description,

    long_description_content_type="type/markdown",

    )
```

## add author, author_email, github link,  to `setup.py` file
```python

from setuptools import setup


setup(

	...,

    author="Geofrey Wanyama",

    author_email="wanyamag17@gmail.com",

    url="git@github.com:G-Geofrey/econometric_package.git",

    zip_safe=False,


    )
```

## add classifiers to setup.py file
```python
from setuptools import setup

setup(
	...,

    zip_safe=False,

    classifers=[ 
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7", 
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent"
    ],


    )
```

## add libarary dependencies to setup.py file
```python
from setuptools import setup

setup(
	...,

    classifers=[ 
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7", 
                "License :: OSI Approved :: MIT License",
                "Operating System :: OS Independent"
    ],

     install_requires=[
     	'pandas>=1.3.4',
		 'numpy>=1.20.3',
		 'matplotlib>=3.4.3',
		 'seaborn>=0.11.2',
		 'statsmodels>=0.12.2'
	]

    )

```

## run the command $ python setup.py bdist_wheel
```bash
python setup.py bdist_whell
```

## install the package locally
```bash
pip install -e . 
```

## test the package on an example code file to check that file works well
```bash
python example.py
```

## run the following command to check that all files are distributed
```bash
python setup.py sdist
```
## run the following command (open the zip file in dist folder)
```bash
tar tzf dist/py_econometrics-0.0.1.tar.gzdis
```

## in case of missing files in zip file above, run the following commands
```bash
pip install check-manifest
check-manifest --create
git add MANIFEST.in
python setup.py sdist

```

## push package to pypi
```bash
pip install twine

twine upload dist/*
```




