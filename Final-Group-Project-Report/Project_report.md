# Prediction of the Outcome of United State Security Clearance Appeals via NLP models
## Introduction
In order to safeguard its national security, the United States requires all personnel working with classified information to obtain a security clearance at one of three security levels: **CONFIDENTIAL**,  **SECRET**, and **TOP SECRET** (**TS**). Additionally, almost all positions with the federal government require a *Public Trust* clearance, making *Public Trust* a de facto "zeroth" clearance level. Security clearances are granted according to guidelines established by the Department of Defense. By law, if an applicant's security clearance is rejected, they must be issued a Statement of Reasons (SOR) which will detail why their application for a clearance was rejected. The applicant can then chose to appeal the decision to the Defense Office of Hearings and Appeals (DOHA).

This project develops NLP models to make predictions if an applicant's security clearance appeal will be rejected or approved depending on "Findings of Fact" in their appeal case. The data extraction from the DOHA's website is described before moving on to descriptions of the two models we will investigate (logistic regression and LSTM netowkrs) and their performance in handling this task. These models will be judged based on the $F_\beta$ score they produce, with an emphasis on preventing false positives. We conclude with an overview of the results for each model, an explainations of limitations of our apporach, and suggestions for improvements.

## Dataset Description
Data was obtained from the DOHA's Website[^1] which contains an archive of appeal cases going back from before 2016. These files contain details of all appeals cases the DOHA hears. These documents are publicly available for download and are included in this repository for convenience since the DOHA website is quite slow. Documents where download going back to 2019, up to the present (2024). In total some 4000 documents are included as part of our data set.

Of interest, all files contain a section known as the *Findings of Fact* (FoF) which detail the evidence accepted into the court before a decision is administered. Information in the FoF are organized under the *security clearance guideline*[^2].

Examples of the 13 guidelines and examples of information related to the guideline include:

- Guideline **B** (Foreign Influence): Details of the relationships with foreign contacts of the applicant, including family, friends, and profession contacts
- Guideline **F** (Financial Consideration): Details on debts held by applicant, accounts in collections, and missed tax payments
- Guideline **H** (Drug Involvement and Substance Misuse): Details of illicit drugs used by the applicant and the time since the last use of the substances in question

Not all FoFs will contain information about all of these topics because FoFs only detail information regarding the *security guideline(s)* the applicant is being questioned on. Thus one could not obtain information on debts held by the applicant if their SOR does at least partially fall under guideline F, beyond the fact that their financial situation was stable enough that it did not worrent mention. We will discuss how this and other lingustic features of FoFs create limitations with our models in the conclusion.

A final note is that we choose to define denial as the positive class. Thus, FoFs which are part of cases where clearance was denied are marked with `1`. This enables the easy use of the recall metric to determine how many clearances that should have been denied were correctly denied. Given the high-risk nature of granting a security clearance, we want to minimize cases where clearance is granted when it shouldn't have.

## Description of models

The there different models be used are:
1. Logistic Regression model with TF-IDF features (Implemented via `sklearn`)
2. Bidirectional LSTM Neutral Net (Implemented via `PyTorch`)

A high-level description of each follows

### Logistic Regression with TF-IDF
Term Frequency Inverse Document Frequency (TF-IDF) are a popular way to make shallow learning models such has logicistic regression and support vector machines reach high levels of accuracy for NLP tasks. The method works by taking every *term* (defined as a word or collection of $n$ words, called generically $n$grams) and getting a percentage of how many times that word appears in a document, hence its name. This is able to cable to convert words or phrase into numerical variables for shallow learning models.

Logistic regression is a classic shallow learning model that involves transforming numeric data to a probabilty of being in one of two binary classes. It learns parameters $\boldsymbol{\beta}$ such that given the TF-IDF matrix $\mathbf{X}$ the rounded sigmaoid function:

$$\text{round}\left(\frac{1}{1+e^{\boldsymbol\beta\cdot\mathbf{X}}}\right)=\text{round}(\sigma(\boldsymbol\beta\cdot\mathbf{X}))$$
will output the correct class (0 for approval or 1 for denial) most of the time.

An advantage of the this model is that it will allow us to detect the relative impact of the presence of difference $n$grams on the probability of approval/denial. This will help us see if some undesirable features are being learned by our model.

### Bidirectional LSTM Network

The LSTM framework was created to address the "vanishing gradient" problem in traditional recurrent neutral networks (RNN). In a neural network, the gradient of the loss function in terms of the weights,  $\nabla\mathcal{L}(\mathbf{W})$, plays an important role in learning since it is used to "guide" the network in the direction of the most optimal parameters. Traditional RNNs have an issue where $|\nabla\mathcal{L}(\mathbf{W})|$ will tend towards 0, causing learning to stop. LSTM networks fix this by introducing a *cell* that will contain information from previous states of learning. Specially, traditional RNNs outputs as $\mathsf{RRN}(\mathbf{X})=\sigma_y(\mathbf{W}\cdot\mathbf{h})$ where $\mathbf{X}$ is the input, $\mathbf{W}$ is the weights learned, $\mathbf{h}$ is the state of the network's hidden layers, and $\sigma_y$. LSTMs instead output $\mathsf{LSTM}(\mathbf{X})=\sigma_y(\mathbf{W}\cdot\mathbf{h})\odot\sigma_c(\mathbf{c})$, which is the output of the RNN scaled element-wise by the cell state. This cell state is maintained separately from the hidden state, preserving information from previous passed along the network.

Bidirectional LSTMs allow more compreiensive cell states by having two networkds run in parallel, one taking information from beginning to end, and one taking information from end to beginning. This makes it so that is does not matter where positionally the dependant of the current context is, it can still be picked up by the network. For example consider the statements
- "In addition to marijuana usage, Applicant used LSD twice while holding a clearance"
- "Applicant admitted to using LSD twice while holding a clearance, as well as several uses or marijuana"

We would want these sentences to have the same effect on our prediction, yet the one that mentions LSD first will likely have a greater effect in a unidirectional network. Bidirectional networks address this flaw.

## Experiment Setup

### Data Preprocessing
Since the hearings are originally in `pdf` format, the `pdfminer` package[^3] is used inorderIn order to prevent legal citation from interfering with the model, we removed all information in parenthesis from the text being fed into the model. This may result in some information being lost.

Contrary to most data cleaning methods, we will not be removing the dollar sign character `$` from our text. This is because dollar amounts come up frequently in the document to indicate debts due by the applicant, thus it is important for models to recongize dollar amounts from other numeric amounts in the data.

For the logistic regression model only, lemmatization via `spacy` will be implemented. Because all information in FoFs took place in the past, tense of verbs should provide no additional information. Since this operation is quite costly, we decided to only preform it as the logistic regression model's pipeline, since the LSTM will be able to understand that the different forms of verbs represent the same lemma via its embedding mechanism.

The dataset ends up being highly imbalanced, with all about a 3:1 ratio between denials and approvals. Since we are mostly optimizing for recall, the danger of models "learning" by denying all applicants is particular high. Thus,  both of our models will be implemented with class weights to prevent overfitting due to class imbalance.

### Class Imbalance
The dataset exhibits a large degree of class imbalance, clearance denials outnumber approvals by 3:1.

In the Logistic Regression model, we used the `class_weight` parameter set to `"balanced"` to address this, however the solution was far more complex for the LSTM model. Class weights seemed to have little effect and would often simply lead to trivial models where every application was approved. Removing classweights would likewise lead to trival models where every application was denied.

The issue appeared to be that the optimizer would get "stuck" in local minima where the best predictions would be uniform denials or approvals due to recieving a few batches in a row that were majority of a single class. Thus, instead of class weights, we define a parameter `train_approve_weight` ($\omega_{a}$) such that given our sampler $\mathcal{S}$

$$\lim_{n\rightarrow\infty}\frac{\text{approvals in} \ n \ \text{samples}}{n}=0.5$$
So that each batch will on average contain an equal number of denials and approvals. Numerical experiments suggest $\omega_a\approx2.7$. Such sampling is done with replacement, which may lead to overfitting. However, as will be explained in the next section, it seems the embedding space of the FoFs is quite "rugged" and it's quite hard to get a model to even *fit* the data, much less overfit. Thus, we opted to remove many safeguards that represent overfitting to ensure convergence of our model.

### Model Judgment

Given the high-risk nature of granting a security clearance, the model will be judged primarily for recall with denial as the positive class. That is to say, the highest priority in judgment of models is to ensure that it does not *grant* a clearance to someone that should have been denied (a false negative, since denial is the positive class). However, we do not entirely want to disregard false positives, thus we will use the $F_\beta$ score with $\beta >> 1$. The $F_\beta$ score if a modification of the $F_1$ score to account for cases where precision and recall are not equally important. It is defined as:

$$F_\beta = \frac{\left(1+\beta^2\right)PR}{\beta^2P+R}=\frac{\left(1+\beta^2\right)t_+}{\left(1+\beta^2\right)t_+ + \beta^2f_-+f_+}$$Where
- $P=\frac{t_+}{f_++t_+}$ is precision
- $R=\frac{t_+}{f_-+t_+}$ is recall
- $t_+,t_-,f_+,f_-$ are the number of true positives (correctly predicted denials), true negatives (correctly predicted approvals), false positives(incorrectly predicted  denials), and false negatives (incorrectly predicted approvals)

This modification of the $F_1$ score captures that recall is $\beta$ times more important than precision if $\beta > 1$ or that precision is $1/\beta$ times more important if $1<\beta<0$. We will set $\beta=3$ for this project.

## Hyperparameter Tuning
### Logistic Regression
Default `sklearn` logistic regression hyperparameters produced quite robust models, thus we opted to not tune hyper parameters beyond excluding one-word $n$grams from our parameters. This was done because many verbs become uninformative without the context of the word immidately next to them. 

### LSTM
The main hyperparameter to optimize was the learning rate $\lambda$. As mentioned before the embedding space for the documents is "rugged" and models are prone to becoming trivial (predicting all applications into one class). This makes optimizer prone to both vanishing and exploding gradients since

- The section of weight space that produce non-trivial models seems to be very "narrow", thus optimizers with high $\lambda$ can oscillate between the section of weight spaces that produces the model that predicts all approvals and the section that predicts all denials
- This ruggedness also makes it so that if the model has a low $\lambda$ and is fed very lob-sided samples for the first few batches, it will "get stuck" in the one of the two sections of weight space that produce trivial models. This will also happen if hyperparameters that counteract overfitting (such as the dropout rate and $L_2$ penalties) are set too high.

Thus we concluded that most effort should be focused on fine-tuning $\lambda$ and choose a very low dropout rate (0.001) to prevent vanishing gradients. Additionally, we add clipping so that $\|\nabla\mathcal{L}\|\leq1$ at each optimization step to prevent exploding gradients causing overshooting into one of the two trivial zones.

## Results

## Summary

### Suggestions for Improvements
In truth, most legal NLP models, especially those that involve predicting legal outcomes, require far greater interpretability to be useful in any real-life application. Thus, many legal judgment models focus on extracting rationale rather than simply making predictions[^4]. This paradigm is particular applicable to security clearance judgements as an applicant can not have a security clearance granted unless they show *all* guideline concerns are mitigated. Thus, a natural extension of this model would focus on feature extraction of the details of each guideline violation to make up its features, rather than the full text of the findings of fact. This would make the model a concatenation of 13 different submodels for each of the security clearance guidelines, returning approval only if all 13 submodels conclude that concerns were mitigated.


[^1]: [https://doha.ogc.osd.mil/Industrial-Security-Program/Industrial-Security-Clearance-Decisions/ISCR-Hearing-Decisions](https://doha.ogc.osd.mil/Industrial-Security-Program/Industrial-Security-Clearance-Decisions/ISCR-Hearing-Decisions)
[^2]: See [https://www.dni.gov/files/NCSC/documents/Regulations/SEAD-4-Adjudicative-Guidelines-U.pdf](https://www.dni.gov/files/NCSC/documents/Regulations/SEAD-4-Adjudicative-Guidelines-U.pdf) for a full description of the security clearance guidelines
[^3]: [https://pdfminersix.readthedocs.io/en/latest/](https://pdfminersix.readthedocs.io/en/latest/)
[^4]: For example, see https://doi.org/10.48550/arXiv.2103.13084
