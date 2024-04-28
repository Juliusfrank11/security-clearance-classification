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

Logistic regression (LR) is a classic shallow learning model that involves transforming numeric data to a probabilty of being in one of two binary classes. It learns parameters $\boldsymbol{\beta}$ such that given the TF-IDF matrix $\mathbf{X}$ the rounded sigmaoid function:

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

Contrary to most data cleaning methods, we will not be removing the dollar sign character `$` from our text. This is because dollar amounts come up frequently in the document to indicate debts due by the applicant, thus it is important for models to recognize dollar amounts from other numeric amounts in the data.

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
- $t_+,t_-,f_+,f_-$ are the number of true positives (correctly predicted denials), true negatives (correctly predicted approvals), false positives(incorrectly predicted denials), and false negatives (incorrectly predicted approvals)

This modification of the $F_1$ score captures that recall is $\beta$ times more important than precision if $\beta > 1$ or that precision is $1/\beta$ times more important if $1<\beta<0$. We will set $\beta=3$ for this project.

## Hyperparameter Tuning
### Logistic Regression
Default `sklearn` parameters for logistic regression produced quite robust models, thus we opted not to fine tune any continuous hyperparameters. However, we did try modifying the minimum and maximum $n$ allowed for $n$grams in our TF-IDF vectorizer, as well as the type of penalty term applied. Lasso ($L_1$) penalty terms could help reduce the impact of unimportant, but common $n$grams.

### LSTM
The main hyperparameter to optimize was the learning rate $\lambda$ and dropout rate $d$. As mentioned before the embedding space for the documents is "rugged" and models are prone to becoming trivial (predicting all applications into one class). This makes optimizer prone to both vanishing and exploding gradients since

- The section of weight space that produce non-trivial models seems to be very "narrow", thus optimizers with high $\lambda$ can oscillate between the section of weight spaces that produces the model that predicts all approvals and the section that predicts all denials
- This ruggedness also makes it so that if the model has a low $\lambda$ and is fed very lob-sided samples for the first few batches, it will "get stuck" in the one of the two sections of weight space that produce trivial models. This will also happen if hyperparameters that counteract overfitting (such as the dropout rate and $L_2$ penalties) are set too high.

Thus, we our first goal was to find $\lambda$ such that models will be non-trival. $\lambda=0.001$ consistently produced non-trivial models, thus only results from models with this learning rate will be included in our results, as all other learning rates tested produced trivial models. As a further safety measure, we used the $\mathsf{AMSGRAD}$ as our optimizer, a varient of $\mathsf{ADAM}$ that is robust to highly non-convex parameter spaces while still including an adaptive learning rate. This allows our model to stay within the non-trivial zone of parameter space.

After discovering a value for $\lambda$, we experimented with adding back a dropout layer since our model can overfit within the non-trivial zone. We test dropout rates of 50%, 10% and 0.1%.

## Results
|  |  |  |  |  |  |  |  |  |  |  |  |
|--|--|--|--|--|--|--|--|--|--|--|--|
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |

### Logistic Regression Results
| $n_{\min}$ | $n_{\max}$ | `penalty` | Accuracy  | Precision | Recall    | $F_1$     | $F_3$     |
|------------|------------|-----------|:---------:|:---------:|:---------:|:---------:|:---------:|
| 1          | 3          | Lasso     | 0.793     | 0.865     | 0.848     | 0.857     | 0.850     |
| 1          | 3          | E-Net     | 0.723     | 0.863     | 0.737     | 0.795     | 0.748     |
| 1          | 3          | Ridge     | 0.735     | **0.885** | 0.732     | 0.801     | 0.745     |
| 1          | 4          | Lasso     | 0.793     | 0.863     | 0.851     | 0.857     | 0.853     |
| 1          | 4          | E-Net     | 0.715     | 0.855     | 0.734     | 0.790     | 0.744     |
| 1          | 4          | Ridge     | 0.721     | 0.875     | 0.720     | 0.790     | 0.733     |
| 2          | 3          | Lasso     | 0.789     | 0.836     | 0.883     | 0.859     | 0.878     |
| 2          | 3          | E-Net     | 0.693     | 0.841     | 0.714     | 0.772     | 0.725     |
| 2          | 3          | Ridge     | 0.694     | 0.844     | 0.712     | 0.773     | 0.724     |
| 2          | 4          | Lasso     | **0.792** | 0.828     | **0.903** | **0.864** | **0.895** |
| 2          | 4          | E-Net     | 0.680     | 0.830     | 0.705     | 0.763     | 0.716     |
| 2          | 4          | Ridge     | 0.674     | 0.829     | 0.697     | 0.757     | 0.709     |

### LSTM Results
| `dropout_rate` | Accuracy  | Precision | Recall    | $F_1$     | $F_3$     |
|----------------|-----------|-----------|-----------|-----------|-----------|
| 0.5            | 0.592     | **0.792** | 0.598     | 0.681     | 0.613     |
| 0.1            | **0.711** | 0.776     | **0.845** | **0.809** | **0.838** |
| 0.001          | 0.658     | 0.782     | 0.738     | 0.759     | 0.742     |

Above are results for both models. In the end, the best LSTM mode ($d=0.1$) is out preformed by every LR model that includes a ridge (`l1_ratio = 0`) penalty term. In fact, the best logistic regression model (2,4,0) outperforms the best LSTM model in every metric. This implies that semantics rarely matter in security clearance descisions, instead, a few "red flags" have large weights. This would explain why the LSTM model with a 50% dropout rate preforms so poorly, it likely drops much of embedding space which contains these "red flag" conotations.

Support for this theory can be found by looking at the $n$grams with the largest positive coefficients (most contribution to denial) in the best LR model, referred to $\mathsf{LR}(2,4,0)$, seen below.

### Largest coefficients amount $n$grams in $\mathsf{LR}(2,4,0)$
|$n$gram|$\beta_i$|
|--|--|
|use marijuana|1.8993601427269533
|applicant claim|1.2127381291381891
|applicant state|1.1958245749981677
|tax year|1.1297414776739618
|applicant admit|1.0334732182654243
|place collection|1.0048662271294424
|answer sor|0.9559628895504945
|documentary evidence|0.9437236054307672
|unresolved sor|0.9425697195292493
|tell investigator|0.8595430325777355
|debt remain|0.8254623879692009
|debt unresolved|0.7934606674303728
|provide evidence|0.7887538330283138
|debt sor|0.7501978263723211
|government investigator|0.7320308804891255
|account place collection|0.7218736670660181
|marijuana use|0.7184971738696724
|account place|0.7115397566459751
|tax return|0.7062929143271278
|make payment|0.6915360339963126

As we can see, $n$grams relating to debt and marijuana use dominate the model. In contrast, the $n$grams with the strongest negative coefficients (contributing to approval) relate to payment of debt and loyalty to the United States and its allies.

### Largest negative coefficients amount $n$grams in $\mathsf{LR}(2,4,0)$

| $n$gram                | $\beta_i$           |
|------------------------|---------------------|
| united states          | -1.9390762444418637 |
| applicant wife         | -1.4023163656679243 |
| united state           | -1.3916602727385616 |
| south korea            | -1.1667595272041076 |
| credibly testify       | -1.1564580517093903 |
| debt resolve           | -1.0672568886122413 |
| monthly payment        | -1.050131772431794  |
| student loan           | -0.8533767686892256 |
| citizen resident       | -0.8482770813321437 |
| applicant credibly     | -0.8213614831050416 |
| evidence documentation | -0.7674057070988334 |
| applicant contact      | -0.7636224962466511 |
| documentation creditor | -0.7345304843700309 |
| applicant husband      | -0.6628623776244902 |
| state applicant        | -0.6050158721672176 |
| immigrate united       | -0.5792400164302883 |
| debt resolve sor       | -0.5664899670881716 |
| applicant work         | -0.5582412875564088 |
| applicant pay          | -0.5552669260689507 |

These tables reveal several limitation of our models

1. The higher coefficient on `"applicant wife"` than `"applicant husband"` implies that the model is learning to favor men compared to women inherently. This is a known issue of many NLP models[^4], and is especially prevalent for a TI-IDF approach where individual words matter much more.
2. Security clearances can be denied for very different reasons and our model doesn't account for that fact. Rarer reasons for security clearance regression will inherently have lower weights applied to the $n$grams (or embedding dimension). This can be seen by the fact that the coefficient for `"child pornography"` and in $\mathsf{LR}(2,4,0)$ is only 0.1693, despite the fact we would expect such a mention to have very harsh consequences for one's chances at getting a clearance. Similarly `"use lsd"` and `"use cocaine"` have a far lower coefficents than `"use marijuana"`, around 0.18 for both. Some problematic $n$grams, such as `"sexual harassment"` even have a negative weight, meaning the model considers their mention as beneficial for the applicant
3. The above issue presents a concerning impact for issues of foreign influence or preference. Foreign influence/perference is itself is a rarer reason for clearance rejection and it is unlikely many countries have plentiful case studies to learn from. Seen below, this can lead to situations where citizen in countries with US adversaries can be viewed favorably. Although it seems that this is the exception, it leads to Iranian citizens having better chances at gaining a clearnace than Israeli citizens[^5].

### Coefficients of `"{country_adj} citizen"` in $\mathsf{LR}(2,4,0)$
| Friendly Country | $\beta_i$ | Hostile Country | $\beta_i$ |
|------------------|-----------|-----------------|-----------|
| Ukraine          | -0.0183   | Russia          | 0.0104    |
| Taiwan           | -0.04091  | China           | 0.0223    |
| Colombia         | 0.0183    | Venezuela       | 0.0075    |
| Israel           | 0.0135    | Iran            | -0.06230  |


This is all to say that the impressive results of the model should be taken with more than a grain of salt. The model will likely underfit in cases where the issues at hand involve uncommon "red flags." Still, it is remarkable that even with these weaknesses, this model would have effectively only let 10% of bad applicants get their clearance. Considering the rather vague guidelines for issuing a clearance and the fact that even to this day many bad actors gain clearances despite the US' best efforts, the preformance of such a simple model remains impressive. But under no circumstances would we suggest that this model could even come close to replacing a careful examination of the facts by a human judge.
## Summary
We created two classes of NLP models to determine if an applicant should have their security clearances granted based on the statements of facts about their case. In the end, we find that the shallow learning LR model outpreforms the LSTM model due to clearances being decided by a small number of "red flags." Our LR model grants clearances to only 10% of applicants that should have been denied, but the coefficients of the model reveal several kinds of undesirable behavior, such as sexist bias and overemphasis on common "red flags." We think that this topic warents further research and provide some suggestions for improvment

### Suggestions for Improvements
In truth, most legal NLP models, especially those that involve predicting legal outcomes, require far greater interpretability to be useful in any real-life application. Thus, many legal judgment models focus on extracting rationale rather than simply making predictions[^6]. This paradigm is particular applicable to security clearance judgements as an applicant can not have a security clearance granted unless they show *all* guideline concerns are mitigated. Thus, a natural extension of this model would focus on feature extraction of the details of each guideline violation to make up its features, rather than the full text of the findings of fact. This would make the model a concatenation of 13 different submodels for each of the security clearance guidelines, returning approval only if all 13 submodels conclude that concerns were mitigated.

Additionally, it would be wise to clean the raw text of characteristics relating to protected characteristics (other than nationality, for obvious reasons). Models using protected characistics to make descisions on legal cases is legally, not to mention morally, unacceptable.

Finally, we suggest pre-training the model on information about security clearance guidelines, specifically in the case of training a transformer on this task. This would help make the model more robust towards rarer "red flags" such as use of hard drugs, sexual crimes, and citizenship from more exotic hostile nations. A good place to start would be to feed the model the *Adjudicative Desk Reference* published by the DoD[^7] that contains numerious case studies of how to determine if an applicant should be granted a clearance.


[^1]: [https://doha.ogc.osd.mil/Industrial-Security-Program/Industrial-Security-Clearance-Decisions/ISCR-Hearing-Decisions](https://doha.ogc.osd.mil/Industrial-Security-Program/Industrial-Security-Clearance-Decisions/ISCR-Hearing-Decisions)
[^2]: See [https://www.dni.gov/files/NCSC/documents/Regulations/SEAD-4-Adjudicative-Guidelines-U.pdf](https://www.dni.gov/files/NCSC/documents/Regulations/SEAD-4-Adjudicative-Guidelines-U.pdf) for a full description of the security clearance guidelines
[^3]: [https://pdfminersix.readthedocs.io/en/latest/](https://pdfminersix.readthedocs.io/en/latest/)
[^4]: See https://doi.org/10.48550/arXiv.2112.14168 for a overview on sexist bias in NLP models.
[^5]: This specific example is somewhat explainable: Israeli spy Jonathan Pollard caused one of the worst leaks of classified information and admitted that he sold his "services" to the Soviet Union. This has lead to increased scrutiny of applicants with ties to Israel. In contrast, many mentions of Iranian citizenship may be referring exiles who had Iranian citizenship before the Islamic Revolution, who are very unlikely to help Iran with their espionage activities.
[^6]: For example, see https://doi.org/10.48550/arXiv.2103.13084
[^7]: Avaliable at https://www.dhra.mil/Portals/52/Documents/perserec/ADR_Version_4.pdf
