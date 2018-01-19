# coding: utf-8

"""
    description: model validation utilities class
    author: Suraj Iyer
"""

import numpy as np
import gc

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _translate_train_sizes, _fit_and_score


def learning_curve(sc, estimator, X, y, groups=None, train_sizes=np.linspace(0.1, 1.0, 5),
                   cv=None, scoring=None, verbose=0, n_jobs=1):
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    # Store it as list as we will be iterating over the list multiple times
    cv_iter = list(cv.split(X, y, groups))

    scorer = check_scoring(estimator, scoring=scoring)

    n_max_training_samples = len(cv_iter[0][0])
    # Because the lengths of folds can be significantly different, it is
    # not guaranteed that we use all of the available training data when we
    # use the first 'n_max_training_samples' samples.
    train_sizes_abs = _translate_train_sizes(train_sizes,
                                             n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))

    train_test_proportions = []
    for train, test in cv_iter:
        for n_train_samples in train_sizes_abs:
            train_test_proportions.append((train[:n_train_samples], test))

    cv_splits = [(train, test) for train, test in train_test_proportions]
    indexed_cv_splits = list(zip(range(len(cv_splits)), cv_splits))
    par_cv_splits = sc.parallelize(indexed_cv_splits, len(indexed_cv_splits))
    X_bc = sc.broadcast(X)
    y_bc = sc.broadcast(y)
    scorer_bc = sc.broadcast(scorer)

    def fun(tup):
        (index, (train, test)) = tup
        local_X = X_bc.value
        local_y = y_bc. value
        local_scorer = scorer_bc.value
        res = _fit_and_score(clone(estimator), local_X, local_y, local_scorer, train, test, 0,
                             parameters=None, fit_params=None, return_train_score=True)
        gc.collect()
        return index, res

    indexed_out0 = dict(par_cv_splits.map(fun).collect())
    out = [indexed_out0[idx] for idx in range(len(cv_splits))]
    del indexed_out0
    gc.collect()
    X_bc.unpersist()
    y_bc.unpersist()
    scorer_bc.unpersist()

    out = np.array(out)
    gc.collect()
    n_cv_folds = out.shape[0] // n_unique_ticks
    out = out.reshape(n_cv_folds, n_unique_ticks, 2)
    gc.collect()

    out = np.asarray(out).transpose((2, 1, 0))
    gc.collect()

    return train_sizes_abs, out[0], out[1]

def validation_curve(sc, estimator, X, y, param_name, param_range, groups=None,
                     cv=None, scoring=None, n_jobs=1, pre_dispatch="all",
                     verbose=0):
    """Validation curve.

    Determine training and test scores for varying parameter values.

    Compute scores for an estimator with different values of a specified
    parameter. This is similar to grid search with one parameter. However, this
    will also compute training scores and is merely a utility for plotting the
    results.

    Read more in the :ref:`User Guide <learning_curve>`.

    Parameters
    ----------
    sc : Spark context
    
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    param_name : string
        Name of the parameter that will be varied.

    param_range : array-like, shape (n_values,)
        The values of the parameter that will be evaluated.

    groups : array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    pre_dispatch : integer or string, optional
        Number of predispatched jobs for parallel execution (default is
        all). The option can reduce the allocated memory. The string can
        be an expression like '2*n_jobs'.

    verbose : integer, optional
        Controls the verbosity: the higher, the more messages.

    Returns
    -------
    train_scores : array, shape (n_ticks, n_cv_folds)
        Scores on training sets.

    test_scores : array, shape (n_ticks, n_cv_folds)
        Scores on test set.

    Notes
    -----
    See :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`

    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)

    param_range = [(train, test, param_name, v) for train, test in cv.split(X, y, groups) for v in param_range]
    indexed_param_range = list(zip(range(len(param_range)), param_range))
    par_param_range = sc.parallelize(indexed_param_range, len(indexed_param_range))
    X_bc = sc.broadcast(X)
    y_bc = sc.broadcast(y)
    scorer_bc = sc.broadcast(scorer)

    def fun(tup):
        (index, (train, test, local_name, local_v)) = tup
        local_X = X_bc.value
        local_y = y_bc.value
        local_scorer = scorer_bc.value
        res = _fit_and_score(clone(estimator), local_X, local_y, local_scorer, train, test, 0,
                             parameters={local_name: local_v}, fit_params=None, return_train_score=True)
        gc.collect()
        return index, res

    indexed_out0 = dict(par_param_range.map(fun).collect())
    out = [indexed_out0[idx] for idx in range(len(param_range))]
    del indexed_out0
    gc.collect()
    X_bc.unpersist()
    y_bc.unpersist()
    scorer_bc.unpersist()

    out = np.asarray(out)
    n_params = len(param_range)
    n_cv_folds = out.shape[0] // n_params
    out = out.reshape(n_cv_folds, n_params, 2).transpose((2, 1, 0))

    return out[0], out[1]
