# movieclassifier/main.py
"""The entry point for loading, training and evaluating the movieclassifier package."""

from pathlib import Path

import mlflow
import numpy as np

from movieclassifier import config, data, eval, train, utils

# Load data
movie_data_fp = Path(config.DATA_DIR, "interim", "movie_data.csv")
doc_stream = utils.stream_docs(movie_data_fp)
size = 1000
X_train, y_train = utils.get_minibatch(doc_stream, size)
X_train, y_train = np.asarray(X_train, dtype=str), np.asarray(y_train, dtype=str)
X_test, y_test = utils.get_minibatch(doc_stream, size=5000)
X_test, y_test = np.asarray(X_test, dtype=str), np.asarray(y_test, dtype=str)


# Preprocess data
X_train = data.hash_vectorize(X_train)
X_test = data.hash_vectorize(X_test)

# Tracking

# Set experiment
mlflow.set_experiment(experiment_name="online-logreg")

with mlflow.start_run() as run:
    # Train model
    clf = train.train(X_train, y_train)

    # Evaluate model
    accuracy = eval.evaluate(clf, X_test, y_test)
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "model")
    print(f"Accuracy: {accuracy}")
    print()
    print("Logged data and model in run: {}".format(run.info.run_id))
