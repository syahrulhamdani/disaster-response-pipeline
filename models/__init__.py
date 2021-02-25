import re

import pandas as pd
from nltk import pos_tag, sent_tokenize, word_tokenize, WordNetLemmatizer
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion


class StartingWithVerb(TransformerMixin):
    """A feature extractor whether a text starts with a verb."""
    def _is_start_with_verb(self, text):
        list_sentences = sent_tokenize(text)
        for sentence in list_sentences:
            tags = pos_tag(tokenize(sentence))
            first_word, first_tag = tags[0]
            if first_word.startswith("V") or (first_word == "RT"):
                return True
        return False

    def fit(self, X, y=None):
        """Fit method which returns instance itself."""
        return self

    def transform(self, X):
        tag = pd.Series(X).apply(self._is_start_with_verb)
        return pd.DataFrame(tag)


def tokenize(text: str):
    """Tokenizes a single text.

    Args:
        text (str): string of text.

    Returns:
        List[str]: list of generated tokens
    """
    lemmatizer = WordNetLemmatizer()
    url_regex = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|'
        '(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    for url in url_regex.findall(text):
        text = text.replace(url, "URLPLACEHOLDER")

    tokens = word_tokenize(text)
    clean_token = [
        lemmatizer.lemmatize(token, pos="v").lower().strip()
        for token in tokens
    ]
    return clean_token


def build_model():
    """Build a model.

    The model is in a pipeline with 2 steps, `features` and `classifier`.
    `features` is a feature union of TF-IDF and `StartingWithVerb` feature
    extractor. The `classifier` used currently is a random forest model.

    Returns:
        A pipeline model.
    """
    params = {
        "features__vectorizer__min_df": [.01, .03],
        "classifier__estimator__min_samples_split": [100, 200]
    }
    model = Pipeline([
        ("features", FeatureUnion([
            ("vectorizer", TfidfVectorizer(
                tokenizer=tokenize, max_df=.6, min_df=.03
            )),
            ("verb", StartingWithVerb())
        ])),
        ("classifier", MultiOutputClassifier(RandomForestClassifier(
            min_samples_split=100, n_estimators=20, random_state=1
        )))
    ])
    return GridSearchCV(model, param_grid=params)
