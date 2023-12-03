[![Bladesight - This package contains helps you follow along with the bladesight tutorials](https://intro-to-btt-using-python-assets.s3.amazonaws.com/bladesight_logo_horizontal_ORIGINAL.jpg)](https://github.com/Bladesight)

------------------------------------------------------------------------

[![PyPI version](https://badge.fury.io/py/bladesight.svg)](https://badge.fury.io/py/bladesight)
[![Downloads](https://static.pepy.tech/badge/bladesight)](https://static.pepy.tech/badge/bladesight)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://img.shields.io/badge/python-3.9+-blue.svg)
_________________

This package contains utilities to help you follow along with the Bladesight tutorials.

## Installation

```bash
pip install bladesight
```

## Usage

### Datasets

All the datasets used in the tutorials are hosted on AWS on a public S3 bucket. This package makes it seamless to download and use these datasets. First import the Datasets object.

```python
from bladesight import Datasets
```

You can list the available datasets by printing the using the `list_datasets` method.

```python
>>> print(Datasets.online_datasets)
['bladesight-datasets/intro_to_btt/intro_to_btt_ch02']
```
You can then load a dataset by selecting the dataset like you would have selected a dictionary key, just by using the 'data/' prefix:

```python
>>> my_dataset = Datasets['data/intro_to_btt/intro_to_btt_ch02']
```
![Example Usage](https://intro-to-btt-using-python-assets.s3.amazonaws.com/load_dataset.gif)


You can then load the table by selecting the table like you would have selected a dictionary key, just by using the 'table/' prefix:

```python 
>>> df_proximity_probe = my_dataset['table/aluminium_blisk_1200_rpm']
```
![Example Usage](https://intro-to-btt-using-python-assets.s3.amazonaws.com/load_table.gif)

The datasets have been downloaded to your local machine. You can find them in the following directory:

```console
~/.bladesight/data
```

### Citations

You'll see an attribution every time you open a dataset. This attribution is a link to the dataset's citation. 

![Example Usage](https://intro-to-btt-using-python-assets.s3.amazonaws.com/citation_example.png)

### Uploading your data

If you'd like your datasets to be accessible through this package, I'd be more than willing to convert it to the appropriate format and upload it.

Please email me at <a href="mailto:dawie.diamond@bladesight.com">dawie.diamond@bladesight.com </a> to get the ball rolling.

## Functions and methods

I include all the functions that are used frequently throughout the tutorials in the bladesight package. This is to avoid cluttering the tutorials with code that is not relevant to the tutorial's objective.

```python
from bladesight.btt.triggering_criteria import threshold_crossing_hysteresis_pos
```

## License

This project is licensed under the terms of the MIT license.