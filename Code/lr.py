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
)

FBETA = 3


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


model = Pipeline(
    [
        ["regex_substitutor", RegexSubstitutor(pattern=r"[^\w\s$-]", replacement="")],
        ["lemmatizer", Lemmatizer("en_core_web_lg")],
        [
            "tfidf",
            TfidfVectorizer(
                ngram_range=(2, 5),
                stop_words="english",
            ),
        ],
        ["lr", LogisticRegression(class_weight="balanced")],
    ]
)

model.fit(train_data.data, train_data.target)

eval_y_true = eval_data.target
eval_y_pred = model.predict(eval_data.data)

print("accuracy","precision","recall","f1",f"f{FBETA}",step="\t")
print(
    accuracy_score(eval_y_true, eval_y_pred),
    precision_score(eval_y_true, eval_y_pred),
    recall_score(eval_y_true, eval_y_pred),
    f1_score(eval_y_true, eval_y_pred),
    fbeta_score(eval_y_true, eval_y_pred, beta=FBETA),
)

import pandas as pd

ngram_to_idx_dict = model['tfidf'].vocabulary_

model_coeffs = model['lr'].coef_

df = pd.DataFrame(model_coeffs,columns=sorted(ngram_to_idx_dict,key= lambda x: ngram_to_idx_dict[x]),index=["Contribution to Rejection"])

df.T.sort_values("Contribution to Rejection").to_csv("ngram_rejection_coef.csv")

print("Coefficients of ngrams saved to ngram_rejection_coef.csv")

print("Displaying Confusion Matrix")

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

ConfusionMatrixDisplay(confusion_matrix(eval_y_true,eval_y_pred,display_labels=eval_data.target_names)).plot()
plt.show()
