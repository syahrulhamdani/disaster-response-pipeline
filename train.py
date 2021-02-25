import argparse
import os
import re
from time import time

import joblib
import pandas as pd
import nltk
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

from models import build_model, tokenize, StartingWithVerb


PARSER_DESC = "Script to build, train, and save model"
DATADIR = "data"
MODEL_DIR = "models"

nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])


def get_arguments():
    parser = argparse.ArgumentParser(description=PARSER_DESC)
    parser.add_argument("database", action="store", type=str,
                        help="Path to database to be loaded")
    parser.add_argument("model", action="store", type=str,
                        help="Model filename used to save the trained model")
    args = parser.parse_args()
    args.database = os.path.join(DATADIR, args.database)
    args.model = os.path.join(MODEL_DIR, args.model)
    return args


def load_data(database_filepath: str):
    """Load data from sql database.

    Args:
        database_filepath (str): path to sql database

    Returns:
        A tuple of messages (pd.Series), labels (pd.DataFrame), and
        list of label names (List).
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster", engine)
    df = df.reset_index(drop=True)
    X = df["message"]
    # drop `child_alone` column since all its value are zeros
    Y = df.drop(columns=["message", "original", "genre", "id", "child_alone"])
    # replace value 2, if exist, to 0 for consistency
    Y = Y.replace({2: 0})
    return X, Y, Y.columns.tolist()


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate trained model with development set.

    Args:
        model (Pipeline): trained model
        X_test (pd.DataFrame): predictors of development or test set
        Y_test (pd.DataFrame): target labels of development or test set
        category_names (List[str]): list of label/category names
    """
    test_prediction = model.predict(X_test)

    eval_filepath = "models/eval.txt"
    if os.path.exists(eval_filepath):
        os.remove(eval_filepath)

    for idx, category in enumerate(category_names):
        print(f"Performance on {category}..")
        print(classification_report(
            Y_test[category], test_prediction[:, idx], zero_division=False
        ))
        with open(eval_filepath, "a") as f:
            f.write(f"Performance on {category}..\n")
            f.write(classification_report(
                Y_test[category], test_prediction[:, idx], zero_division=False
            ))
            f.write("\n")

    print(f"Saved evaluation resutls: {eval_filepath}")


def save_model(model, model_filepath):
    """Save model in model_filepath."""
    joblib.dump(model, model_filepath)


def main():
    args = get_arguments()

    print('Loading data...\n    DATABASE: {}'.format(args.database))
    X, Y, category_names = load_data(args.database)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1
    )
    print("Will train with {} training data and validate with "
          "{} validation data".format(
              X_train.shape[0], X_test.shape[0]
          ))

    print('Building model...')
    model = build_model()

    print('Training model...')
    start = time()
    model.fit(X_train, Y_train)
    print('Done training in {:.3f}s'.format(time() - start))

    # use the best estimator only
    print("Best parameters..")
    print(model.best_params_)
    model = model.best_estimator_

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(args.model))
    save_model(model, args.model)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
