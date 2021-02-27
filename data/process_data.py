import argparse
import os

import pandas as pd
from sqlalchemy import create_engine


PARSER_DESC = """
Data Processor to perform extract, transform, load steps.
"""
DATADIR = "data"


def get_arguments() -> argparse.Namespace:
    """Get arguments passed.

    These arguments are required for ETL steps.

    Returns:
        argparse.Namespace: a namespace for consists of extracted arguments
    """
    parser = argparse.ArgumentParser(description=PARSER_DESC)
    parser.add_argument("message", action="store", type=str,
                        help="path to message data")
    parser.add_argument("category", action="store", type=str,
                        help="path to category data")
    parser.add_argument("database", action="store", type=str,
                        help="path to database file to save processed data")
    args = parser.parse_args()
    args.message = os.path.join(DATADIR, args.message)
    args.category = os.path.join(DATADIR, args.category)
    args.database = os.path.join(DATADIR, args.database)
    return args


def load_data(
        messages_filepath: str,
        categories_filepath: str
    ) -> pd.DataFrame:
    """Load and combine necessary data.

    Args:
        message_filepath (str): path to message data
        categories_filepath (str): path to categories data

    Returns:
        pd.DataFrame: combined messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset.

    Args:
        df (pd.DataFrame): dataframe

    Returns:
        pd.DataFrame: cleaned data
    """
    df_categories = df.categories.copy()
    df_categories = df_categories.str.split(";", expand=True)
    category_names = df_categories.loc[0].apply(lambda s: s.split("-")[0])
    df_categories.columns = category_names
    df_categories = df_categories.applymap(lambda s: int(s.split("-")[1]))
    df_categories = df_categories.replace({2: 0})

    df = df.drop(columns="categories")
    df = pd.concat([df, df_categories], axis=1)
    df = df.drop_duplicates()
    return df


def save_data(df: pd.DataFrame, database_filename: str):
    """Save cleaned data as in a SQL database.

    Args:
        df (pd.DataFrame): clean data
        database_filename (str): database name
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("disaster", engine, index=False, if_exists="replace")


def main():
    args = get_arguments()
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(args.message, args.category))
    df = load_data(args.message, args.category)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...\n    DATABASE: {}'.format(args.database))
    save_data(df, args.database)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()
