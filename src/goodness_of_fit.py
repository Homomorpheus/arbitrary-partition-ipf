"""
Helper functions for computing the goodness of fit
using the G² statistic.
"""

import numpy as np
import scipy as sc
import warnings


def deviance(data: np.ndarray, estimator: np.ndarray):
    # deviance G²
    return 2 * np.sum(data * np.log(data/estimator))

def deviance_p_value(data: np.ndarray, estimator: np.ndarray, df: int):
    # p-value in favor of the model
    if np.any(estimator < 5):
        raise RuntimeWarning("chi-square approximation not precise")
    dev = deviance(data, estimator)
    return 1 - sc.stats.chi2.cdf(dev, df)



def approx_deviance_nan_masked_subtable_hom_assoc_3d(data, estimator, marginals):
    # approximate deviance and p-value of estimator for nan-padded data

    # step 1: compute deviance to sub-table
    sub_dev = 0
    subtable_size = 0

    it = np.nditer(data, flags=["multi_index"])
    for _ in it:
        index = it.multi_index
        if np.isnan(data[index]):
            continue
        elif estimator[index] != 0:
            subtable_size += 1
            if data[index] != 0:
                sub_dev += 2 * data[index] * np.log(data[index] / estimator[index])

    approx_dev = sub_dev / subtable_size * estimator.size

    # step 2: compute degrees of freedom of full table,
    # for "homogeneous association" (last before saturated model)
    # à la method a from Bishop, Fienberg, Holland (1975) p. 114f
    df = 0

    df += (estimator.shape[0] - 1) * (estimator.shape[1] - 1) * (estimator.shape[2] - 1)

    # subtract number of "elementary cells with zero estimate"
    df -= np.sum(np.isclose(estimator, 0))

    # add number of parameters that "cannot be estimated"
    for marginal in marginals:
        df += np.sum(np.isclose(marginal.value, 0))
    # from 1d marginals a, b, c
    a = np.sum(estimator, axis=(1, 2))
    b = np.sum(estimator, axis=(0, 2))
    c = np.sum(estimator, axis=(0, 1))
    for marginal in (a, b, c):
        df += np.sum(np.isclose(marginal, 0))

    assert not np.isclose(np.sum(estimator), 0.)


    if np.any(estimator < 5):
        warnings.warn(f"chi-square approximation not precise, approx. deviance is {approx_dev}")
    p_val = 1 - sc.stats.chi2.cdf(approx_dev, df=df)

    return approx_dev, p_val, df


def approx_deviance_nan_masked_subtable(data, estimator, df):
    # approximate deviance and p-value of estimator for nan-padded data

    sub_dev = 0
    subtable_size = 0

    it = np.nditer(data, flags=["multi_index"])
    for _ in it:
        index = it.multi_index
        if np.isnan(data[index]):
            continue
        elif estimator[index] != 0:
            subtable_size += 1
            if data[index] != 0:
                sub_dev += 2 * data[index] * np.log(data[index] / estimator[index])

    approx_dev = sub_dev / subtable_size * estimator.size

    assert not np.isclose(np.sum(estimator), 0.)


    if np.any(np.logical_and(estimator < 5, estimator > 0)):
        warnings.warn(f"chi-square approximation not precise, approx. deviance is {approx_dev}")
    p_val = 1 - sc.stats.chi2.cdf(approx_dev, df=df)

    return approx_dev, p_val
