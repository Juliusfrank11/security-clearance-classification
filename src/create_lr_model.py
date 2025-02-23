
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from simplemma import lemmatize

def preprocess_text(text: str):
    gender_neutralizing_dict = {
        "he": "applicant",
        "she": "applicant",
        "husband": "spouse",
        "wife": "spouse",
    }
    
    text_list = text.lower().split()
    for i, word in enumerate(text_list):
        if word in gender_neutralizing_dict.keys():
            text_list[i] = gender_neutralizing_dict[word]

    return " ".join([lemmatize(word, lang="en") for word in text_list if len(word) > 2])



def create_lr_model(guideline: str, test_size: float = 0.2):
    try:
      data = pd.read_csv(
          f"data\\formal_finding_results_guideline_{guideline.upper()}.csv"
      )
    except FileNotFoundError:
      data = pd.read_csv(
          f"src\\data\\formal_finding_results_guideline_{guideline.upper()}.csv"
      )

    train_data, test_data = train_test_split(data, test_size=test_size)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 3),
                    stop_words="english",
                    preprocessor=preprocess_text,
                    min_df=0.01,
                    max_df=0.9,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", penalty='l2', solver="liblinear"
                ),
            ),
        ]
    )

    model.fit(train_data["text"], train_data["label"])

    return model, train_data, test_data