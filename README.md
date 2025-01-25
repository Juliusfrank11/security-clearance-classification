# security-clearance-classification
Webapp for predicting US security clearance appeal outcomes.
## Problem Description
To safeguard its national security, the United States requires all personnel working with classified information to obtain a security clearance at one of three security levels: **CONFIDENTIAL**, **SECRET**, and **TOP SECRET** (**TS**). Additionally, almost all positions within the federal government require a *Public Trust* clearance, effectively making *Public Trust* a de facto "zeroth" clearance level. Security clearances are granted according to guidelines established by the Department of Defense. By law, if an applicant's security clearance is rejected, they must be issued a Statement of Reasons (SOR) detailing why their clearance application was rejected. The applicant can then choose to appeal the decision to the Defense Office of Hearings and Appeals (DOHA). These appeal cases are posted (with personal identifying information removed) on the DOHA's Website.

This appeal process can be extremely costly, with one source citing a starting figure of [\$2,500](https://news.clearancejobs.com/2021/02/12/when-to-hire-a-security-clearance-lawyer-and-what-legal-fees-to-expect/). Baring in mind that most appeal cases are rejected, it would be significantly beneficial for applicants to know if they stand a good chance of winning their appeal before paying this cost. 

## Proposed Solution

Luckily, from a modeling prospective, security clearance appeals present a simple problem compared to many legal judgement prediction models (LJP). In particular, statue requires one of 13 security clearance guidelines to be cited in the SOR issued to the applicant. This greatly reduces the complexity of LJP tasks by narrowing the possible reasons for an outcome. Additionally: appeals cases are handled by each guideline: meaning legal judgement is just the result of the logical `all` of each individual allogation: an applicant must mitigate all allogated security concerns to win the appeal. This structure allows for dividing the LJP task into discrete subtasks.

This project aims to take advantage of this unique element of security clearance appeals by creating LJP models focused on explainablity. This would be used to help inform potential applicants of their likelihood of winning an appeals case before investing into an anttorny to argue their case. This is not meant to be used to replace judges as a provider of legal judgement or lawyers as a provider as legal advice: rather, it is an informed guess at an applicant's chances before the appeals process even begins.

As a proof of concept, this project will create a web UI where applicants can enter details stated on their SOR and get a predictive statement of their chances of winning an appeal. The application will have inputs for general information such as the applicant's work experience and five dedicated fields for information related tthe most common guidelines mentioned in appeal cases:

1. Guideline B: Foreign Influence
2. Guideline E: Personal Conduct
3. Guideline F: Financial Considerations
4. Guideline H: Drug Involvement
5. Guideline J: Criminal Conduct

## Tech Stack
The code for the application will be written exclusively in Python and use the following packages:
- `pdfminer` used for reading pdf data
- `sklearn` for developing shallow models for guidelines B, F, H, and J.
- `llama_index` used for feature extraction and creation for shallow models
- `sentence-tranformers` for development of deep model for guideline E
- `streamlit` for display of the web application