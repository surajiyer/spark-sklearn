# Scikit-learn integration package for Apache Spark

This package contains some tools to integrate the [Spark computing framework](http://spark.apache.org/) with the popular [scikit-learn machine library](http://scikit-learn.org/stable/). Among other tools:
 - train and evaluate multiple scikit-learn models in parallel. It is a distributed analog to the [multicore implementation](https://pythonhosted.org/joblib/parallel.html) included by default in [scikit-learn](http://scikit-learn.org/stable/).
 - convert Spark's Dataframes seamlessly into numpy `ndarray`s or sparse matrices.
 - (experimental) distribute Scipy's sparse matrices as a dataset of sparse vectors.

It focuses on problems that have a small amount of data and that can be run in parallel.
- for small datasets, it distributes the search for estimator parameters (`GridSearchCV` in scikit-learn), using Spark,
- for datasets that do not fit in memory, we recommend using the [distributed implementation in Spark MLlib](https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html).

  > NOTE: This package distributes simple tasks like grid-search cross-validation. It does not distribute individual learning algorithms (unlike Spark MLlib).

**Difference with the [sparkit-learn project](https://github.com/lensacom/sparkit-learn)** The sparkit-learn project aims at a comprehensive integration between Spark and scikit-learn. In particular, it adds some primitives to distribute numerical data using Spark, and it reimplements some of the most common algorithms found in scikit-learn.

## License

This package is released under the Apache 2.0 license. See the LICENSE file.

## Installation

This package is available on PYPI:

	pip install spark-sklearn

This project is also available as as [Spark package](http://spark-packages.org/package/databricks/spark-sklearn).

The developer version has the following requirements:
 - scikit-learn 0.19.1 has been tested.
 - Spark >= 2.1.0. Spark may be downloaded from the [Spark official website](http://spark.apache.org/). In order to use this package, you need to use the pyspark interpreter or another Spark-compliant python interpreter. See the [Spark guide](https://spark.apache.org/docs/latest/programming-guide.html#overview) for more details.
 - [nose](https://nose.readthedocs.org) (testing dependency only)
 - Pandas, if using the Pandas integration or testing. Pandas==0.18 has been tested.

If you want to use a developer version, you just need to make sure the `python/` subdirectory is in the `PYTHONPATH` when launching the pyspark interpreter:

	PYTHONPATH=$PYTHONPATH:./python:$SPARK_HOME/bin/pyspark

__Running tests__ You can directly run tests:

  cd python && ./run-tests.sh

This requires the environment variable `SPARK_HOME` to point to your local copy of Spark.

## Example

Here is a simple example that runs a grid search with Spark. See the [Installation](#installation) section on how to install the package.

```python
from sklearn import svm, grid_search, datasets
from spark_sklearn import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(sc, svr, parameters)
clf.fit(iris.data, iris.target)
```

This classifier can be used as a drop-in replacement for any scikit-learn classifier, with the same API.

## Documentation

[API documentation](http://databricks.github.io/spark-sklearn-docs) is currently hosted on Github pages. To
build the docs yourself, see the instructions in [docs/README.md](https://github.com/databricks/spark-sklearn/tree/master/docs).

## Changelog

- 2015-12-10 First public release (0.1)
- 2016-08-16 Minor release (0.2.0):
   1. the official Spark target is Spark 2.0
   2. support for keyed models
- 2017-09-20 Minor release (0.2.2):
   1. The official Spark target is Spark >= 2.1
- 2017-09-29 Minor release (0.2.3):
   1. Fixes spark-package build of spark-sklearn.
- 2018-01-19 Minor release (0.2.3):
   1. Updated grid_search to support scikit-learn==0.19 which includes many bug fixes over older scikit versions
   2. Added support for distributing calculation of learning curve and validation curve over spark.
