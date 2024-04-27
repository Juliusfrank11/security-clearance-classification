import re
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_files
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)
import pickle

FBETA = 3
NGRAM_MINS = [1, 2]
NGRAM_MAXS = [3, 4]
L1_RATIOS = [0, 0.5, 1]


class RegexSubstitutor(BaseEstimator, TransformerMixin):
    def __init__(self, pattern, replacement):
        self.pattern = pattern
        self.replacement = replacement

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            re.sub(
                self.pattern, self.replacement, text.decode(encoding="utf-8")
            ).lower()
            for text in X
        ]


class Lemmatizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = spacy.load(model)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [" ".join([token.lemma_ for token in self.model(text)]) for text in X]


train_data = load_files(r"data\finding_of_fact\train")
eval_data = load_files(r"data\finding_of_fact\eval")

for NGRAM_MIN in NGRAM_MINS:
    for NGRAM_MAX in NGRAM_MAXS:
        for L1_RATIO in L1_RATIOS:
            model = Pipeline(
                [
                    [
                        "regex_substitutor",
                        RegexSubstitutor(pattern=r"[^\w\s$-]", replacement=""),
                    ],
                    ["lemmatizer", Lemmatizer("en_core_web_lg")],
                    [
                        "tfidf",
                        TfidfVectorizer(
                            ngram_range=(NGRAM_MIN, NGRAM_MAX),
                            stop_words="english",
                        ),
                    ],
                    [
                        "lr",
                        LogisticRegression(
                            class_weight="balanced",
                            solver="saga",
                            penalty="elasticnet",
                            l1_ratio=L1_RATIO,
                        ),
                    ],
                ]
            )

            model.fit(train_data.data, train_data.target)

            eval_y_true = eval_data.target
            eval_y_pred = model.predict(eval_data.data)

            accuracy = accuracy_score(eval_y_true, eval_y_pred)
            precision = precision_score(eval_y_true, eval_y_pred)
            recall = recall_score(eval_y_true, eval_y_pred)
            f1 = f1_score(eval_y_true, eval_y_pred)
            fbeta = fbeta_score(eval_y_true, eval_y_pred, beta=FBETA)
            tn, fp, fn, tp = confusion_matrix(eval_y_true, eval_y_pred).ravel()

            model_suffex = "-".join(
                [str(s).replace(".", "") for s in [NGRAM_MIN, NGRAM_MAX, L1_RATIO]]
            )
            with open("logistic_regression-" + model_suffex, "wb") as f:
                pickle.dump(model, f)
            with open("lr_model_results.csv", "a") as f:
                f.write(
                    "\n"
                    + ",".join(
                        [
                            str(s)
                            for s in [
                                NGRAM_MIN,
                                NGRAM_MAX,
                                L1_RATIO,
                                accuracy,
                                precision,
                                recall,
                                f1,
                                fbeta,
                                tn,
                                fp,
                                fn,
                                tp,
                            ]
                        ]
                    )
                )
