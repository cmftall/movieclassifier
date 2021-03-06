# movieclassifier/data.py
"""Implement the necessary data pre-processing."""

import re
from typing import Any, List

from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


stop = stopwords.words("english")


def tokenizer(text: str) -> List[str]:
    """Split text data word by word.

    Args:
        text (str): the text to split

    Returns:
        List[str]: list of words
    """
    text = re.sub("<[^>]*>", "", text)
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\\)|\\(|D|P)", text.lower())
    text = re.sub("[\\W]+", " ", text.lower()) + " ".join(emoticons).replace("-", "")
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def hash_vectorize(
    X: np.ndarray,
    decode_error: str = "ignore",
    n_features: int = 2 ** 21,
    tokenizer: Any = tokenizer,
    preprocessor: Any = None,
) -> np.ndarray:
    """Hash vectorize the data.

    Args:
        X (np.ndarray): a numpy ndarray
        decode_error (str): [description]. Defaults to "ignore".
        n_features (int): [description]. Defaults to 2**21.
        tokenizer (Any): [description]. Defaults to tokenizer.
        preprocessor (Any): [description]. Defaults to None.

    Returns:
        HashingVectorizer: [description]
    """
    vect = HashingVectorizer(
        decode_error=decode_error,
        n_features=n_features,
        preprocessor=preprocessor,
        tokenizer=tokenizer,
    )
    X_vect = vect.transform(X)
    return X_vect
