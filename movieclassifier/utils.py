# tagifai/utils.py
"""Utility functions."""

from pathlib import Path
from typing import Iterator, List, Options, Tuple


def stream_docs(path: Path) -> Iterator[Tuple[str, int]]:
    """Reads in and return one document at a time.

    Args:
        path (Path): file path

    Yields:
        Iterator[Tuple[str, int]]: the review text and the corresponding class label

    Example:
        >>> from movieclassifier import utils
        >>> doc = next(utils.stream_docs(path='data/interim/movie_data.csv'))
        >>> bool(doc)
    """
    with open(path, "r") as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def get_minibatch(
    doc_stream: Iterator[Tuple[str, int]], size: int
) -> Tuple[Options[List[str]], Options[List[int]]]:
    """Give a particular number of documents specified by the size parameter.

    Args:
        doc_stream (Iterator[tuple[str, int]]): a document stream
        size (int): number of documents ot return

    Returns:
        Tuple[Options[List[str]], Options[List[int]]]:
    """
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y
