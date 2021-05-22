# movieclassifier/eval.py
"""Implement the evaluation methods."""

import numpy as np
from sklearn.base import BaseEstimator


def evaluate(clf: BaseEstimator, X: np.ndarray, y: np.ndarray) -> float:
    """Compute the accuracy of te classifier.

    Args:
        clf (BaseEstimator): a classifier
        X (np.ndarray): a numpy ndarray
        y (np.ndarray): a numpy ndarray

    Returns:
        float: te accuracy of te classifier
    """
    return clf.score(X, y)
