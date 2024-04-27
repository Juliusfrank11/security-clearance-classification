import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin


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

model_name = "logistic_regression-2-4-0"
with open(model_name, "rb") as f:
    model = pickle.load(f)

ngram_to_idx_dict = model["tfidf"].vocabulary_

model_coeffs = model["lr"].coef_

df = pd.DataFrame(
    model_coeffs,
    columns=sorted(ngram_to_idx_dict, key=lambda x: ngram_to_idx_dict[x]),
    index=["Contribution to Rejection"],
)

df.T.sort_values("Contribution to Rejection").to_csv("ngram_rejection_coef.csv")

print("Coefficients of ngrams saved to ngram_rejection_coef.csv")
