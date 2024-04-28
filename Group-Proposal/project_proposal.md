# Prediction of the Outcome of United States Security Clearance Appeals via NLP Models
## Problem Description
To safeguard its national security, the United States requires all personnel working with classified information to obtain a security clearance at one of three security levels: **CONFIDENTIAL**, **SECRET**, and **TOP SECRET** (**TS**). Additionally, almost all positions within the federal government require a *Public Trust* clearance, effectively making *Public Trust* a de facto "zeroth" clearance level. Security clearances are granted according to guidelines established by the Department of Defense. By law, if an applicant's security clearance is rejected, they must be issued a Statement of Reasons (SOR) detailing why their clearance application was rejected. The applicant can then choose to appeal the decision to the Defense Office of Hearings and Appeals (DOHA). These appeal cases are posted (with personal identifying information removed) on the DOHA's Website.

This project will develop NLP models to predict whether an applicant's security clearance appeal will be rejected or approved based on the "Findings of Fact" in their appeal case. This will be obtained from the aforementioned DOHA website.

## Models Used
Three different models will be used: 
1. Logistic Regression model with TF-IDF features (Implemented via `sklearn`)
2. Bidirectional LSTM Neural Net (Implemented via `PyTorch`)

### Other Packages Used
In addition to the base Python packages, the following packages will also be used:
- `pdfminer` will be used to extract text from the PDFs of the appeal cases. This includes the *Findings of Fact* and the *Conclusion* section which will be parsed to determine if clearance was rejected or accepted.
- `spacy` will be used for lemmatization in the logistic regression model.

## Model Judgment
Given the high-risk nature of granting a security clearance, the model will be judged primarily for recall with denial as the positive class. That is to say, the highest priority in evaluating models is to ensure that it does not *grant* a clearance to someone who should have been denied (a false negative, since denial is the positive class). However, we do not entirely want to disregard false positives, thus we will use the $F_\beta$ score with $\beta >> 1$. The $F_\beta$ score is a modification of the $F_1$ score to account for cases where precision and recall are not equally important. It is defined as:

$$F_\beta = \frac{\left(1+\beta^2\right)PR}{\beta^2P+R}=\frac{\left(1+\beta^2\right)t_+}{\left(1+\beta^2\right)t_+ + \beta^2f_-+f_+}$$

Where:
- $P=\frac{t_+}{f_++t_+}$ is precision.
- $R=\frac{t_+}{f_-+t_+}$ is recall.
- $t_+,t_-,f_+,f_-$ are the number of true positives (correctly predicted denials), true negatives (correctly predicted approvals), false positives (incorrectly predicted denials), and false negatives (incorrectly predicted approvals).

This modification of the $F_1$ score captures that recall is $\beta$ times more important than precision if $\beta > 1$ or that precision is $1/\beta$ times more important if $0<\beta<1$. We will set $\beta=3$ for this project.
