[![Python Versions](https://img.shields.io/pypi/pyversions/cleopatra.png)](https://img.shields.io/pypi/pyversions/cleopatra)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://serapeum-org.github.io/cleopatra/latest/)
[![codecov](https://codecov.io/github/serapeum-org/cleopatra/branch/main/graph/badge.svg?token=gHxH7ljIC3)](https://codecov.io/github/serapeum-org/cleopatra)

![GitHub last commit](https://img.shields.io/github/last-commit/serapeum-org/cleopatra)
![GitHub forks](https://img.shields.io/github/forks/serapeum-org/cleopatra?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/serapeum-org/cleopatra?style=social)


Current release info
====================

| Name                                                                                                                   | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Version                                                                                                                                                                                                                                                                                                                                                       | Platforms                                                                                                                   |
|------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| [![Conda Recipe](https://img.shields.io/badge/recipe-cleopatra-green.svg)](https://anaconda.org/conda-forge/cleopatra) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/cleopatra.svg)](https://anaconda.org/conda-forge/cleopatra) [![Downloads](https://pepy.tech/badge/cleopatra)](https://pepy.tech/project/cleopatra) [![Downloads](https://pepy.tech/badge/cleopatra/month)](https://pepy.tech/project/cleopatra)  [![Downloads](https://pepy.tech/badge/cleopatra/week)](https://pepy.tech/project/cleopatra)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/cleopatra?color=blue&style=flat-square) ![GitHub all releases](https://img.shields.io/github/downloads/serapeum-org/cleopatra/total) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/cleopatra.svg)](https://anaconda.org/conda-forge/cleopatra) [![PyPI version](https://badge.fury.io/py/cleopatra.svg)](https://badge.fury.io/py/cleopatra) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/cleopatra/badges/version.svg)](https://anaconda.org/conda-forge/cleopatra) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/cleopatra.svg)](https://anaconda.org/conda-forge/cleopatra) |

cleopatra - matplotlib utility package
=====================================================================
**cleopatra** is a Python package providing fast and flexible way to build visualize data using matplotlib. it
provides functionalities to handle 3D arrays and perform various operations on them,
such as plotting, animating, and displaying the array. it also provides functionalities for creating statistical plots,


Main Features
-------------
The `Array` class has the following functionalities:
- Initialize an array object with the provided parameters.
- Plot the array with optional parameters to customize the appearance and display cell values.
- Animate the array over time with optional parameters to customize the animation speed and display points.
- Display the array with optional parameters to customize the appearance and display point IDs.

The `statistical_glyph` module provides a class for creating statistical plots, specifically histograms. The class, 
`StatisticalGlyph`, is designed to handle both 1D (single-dimensional) and 2D (multi-dimensional) data.



Installing cleopatra
===============

Installing `cleopatra` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge cleopatra
```

It is possible to list all the versions of `cleopatra` available on your platform with:

```
conda search cleopatra --channel conda-forge
```

## Install from GitHub
to install the last development to time, you can install the library from GitHub
```
pip install git+https://github.com/serapeum-org/cleopatra
```

## pip
to install the last release, you can easily use pip
```
pip install cleopatra
```
