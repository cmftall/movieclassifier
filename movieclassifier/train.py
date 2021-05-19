# movieclassifier/train.py
"""Implement the training method."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier


def train(X: np.ndarray, y: np.ndarray) -> BaseEstimator:
    """Fit model on training data.

    Args:
        X (np.ndarray): [description]
        y (np.ndarray): [description]

    Returns:
        BaseEstimator: a trained estimator
    """
    clf = SGDClassifier(loss="log", random_state=1, n_iter_no_change=1)
    return clf.fit(X, y)
