# Prediction of the Outcome of United State Security Clearance Appeals via NLP models
## Problem Description
In order to safeguard its national security, the United States requires all personnel working with classified information to obtain a security clearance at one of three security levels: **CONFIDENTIAL**,  **SECRET**, and **TOP SECRET** (**TS**). Additionally, almost all positions with the federal government require a *Public Trust* clearance, making *Public Trust* a de facto "zeroth" clearance level. Security clearances are granted according to guidelines established by the Department of Defense. By law, if an applicant's security clearance is rejected, they must be issued a Statement of Reasons (SOR) which will detail why their application for a clearance was rejected. The applicant can then chose to appeal the decision to the Defense Office of Hearings and Appeals (DOHA). These appeal cases are posted (with personal identifying information removed) on the [DOHA's Website](https://doha.ogc.osd.mil/Industrial-Security-Program/Industrial-Security-Clearance-Decisions/ISCR-Hearing-Decisions).

This project will develop NLP models to make predictions if an applicant's security clearance appeal will be rejected or approved depending on "Findings of Fact" in their appeal case. This will be obtained from the forementioned DOHA website.

## Models Used

Three different models will be used: 
1. Logistic Regression model with TF-IDF features (Implemented via `sklearn`)
2. Bidirectional LSTM Neutral Net (Implemented via `PyTorch`)

### Other Packages Used
Other than the base Python packages and the packages, the following packages will also be used:
- `pdfminer` will be used to extract text from the PDFs of the appeal cases. This includes the *Findings of Fact* and the *Conclusion* section which will be parsed to determine if clearance was rejected or accepted
- `spacy` will be used for lemmatization in the logistic regression model
## Model Judgment
Given the high-risk nature of granting a security clearance, the model will be judged primarily for recall with denial as the positive class. That is to say, the highest priority in judgment of models is to ensure that it does not *grant* a clearance to someone that should have been denied (a false negative, since denial is the positive class). However, we do not entirely want to disregard false positives, thus we will use the $F_\beta$ score with $\beta >> 1$. The $F_\beta$ score if a modification of the $F_1$ score to account for cases where precision and recall are not equally important. It is defined as:

$$F_\beta = \frac{\left(1+\beta^2\right)PR}{\beta^2P+R}=\frac{\left(1+\beta^2\right)t_+}{\left(1+\beta^2\right)t_+ + \beta^2f_-+f_+}$$Where
- $P=\frac{t_+}{f_++t_+}$ is precision
- $R=\frac{t_+}{f_-+t_+}$ is recall
- $t_+,t_-,f_+,f_-$ are the number of true positives (correctly predicted denials), true negatives (correctly predicted approvals), false positives(incorrectly predicted  denials), and false negatives (incorrectly predicted approvals)

This modification of the $F_1$ score captures that recall is $\beta$ times more important than precision if $\beta > 1$ or that precision is $1/\beta$ times more important if $1<\beta<0$. We will set $\beta=3$ for this project.