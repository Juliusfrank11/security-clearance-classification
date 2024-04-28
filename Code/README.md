# File descriptions
- `data`: Findings of fact, conclusions, and raw `pdf`s of hearings used to train models
- `LSTM.py`: Model Creation script for LSTM models
- `lstm_model_results.csv`: Results from different hyperparameters for LSTM modes
- `make_model_coef_csv.py`: Used to extract coefficient of $n$grams from logistic regression modes
- `create_dataset_from_pdf.py`: create training and evaluation datasets from pdfs in `data/hearings`
- `lr.py`: Model creation script for logistic regression models.
- `ngram_rejection_coefs.zip`: Zipped results of `make_model_coef_csv.py` for models $\mathsf{LR}(2,4,0)$ (`_best`) and $\mathsf{LR}(1,4,0)$ (`_all`). This notation is the (`ngram_min`,`ngram_max`,`l1_ratio`) of the model. Too large to include individually in this repository.
