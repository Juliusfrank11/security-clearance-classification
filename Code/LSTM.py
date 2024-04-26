import os
import re
import numpy as np

from typing import List, Tuple
from datasets import Dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import DataLoader, WeightedRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 10
SAMPLES_PER_EPOCH = 100
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.1
LOAD_MODEL = ""
FBETA = 3

def load_text_files(directory: str) -> Tuple[List[str], List[str]]:
    """Gathers content of files of one of the subfolder in the `data` folder into two lists

    Args:
        directory (str): the folder path to scan, `data/finding_of_fact/{train|eval}/{approve|deny}

    Returns:
        Tuple[List[str], List[str]]: first element is the text content of the findings of fact, second is the file path used
    """
    texts = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                texts.append(re.sub(r"[^\w\s$]", "", file.read().lower()))
            labels.append(directory)  # Assuming the directory name is the class label
    return texts, labels

if not LOAD_MODEL:
    # Data Loading
    approve_texts, approve_labels = load_text_files(r"data/finding_of_fact/train/approve")
    deny_texts, deny_labels = load_text_files(r"data/finding_of_fact/train/deny")
    texts = approve_texts + deny_texts
    labels = np.array(
        list(map(lambda x: 0 if x.endswith("approve") else 1, approve_labels + deny_labels))
    )

    dataset = Dataset.from_dict({"text": texts, "label": labels}).train_test_split(
        test_size=0.1
    )
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    # Creating vocabulary
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    import torchtext

    vocab_freq_dict = {"<unknown>": 1}
    for text in deny_texts + approve_texts:
        for word in text.split():
            word = word.lower()
            if word not in ENGLISH_STOP_WORDS:
                if word not in vocab_freq_dict.keys():
                    vocab_freq_dict[word] = 1
                else:
                    vocab_freq_dict[word] += 1

    vocab = torchtext.vocab.vocab(vocab_freq_dict)
    vocab.set_default_index(0)  # maps to <unknown>

    # creating data loaders
    train_labels_array = np.array(train_dataset["label"])
    test_labels_array = np.array(test_dataset["label"])
    # train_approve_weight should be set such that the limiting portion of approvals in all samples is 0.5
    train_approve_weight = 2.7
    train_deny_weight = 1
    test_approve_weight = 1
    test_deny_weight = 1

    train_sampler = WeightedRandomSampler(
        weights=[
            train_deny_weight if l > 0 else train_approve_weight for l in train_labels_array
        ],
        num_samples=BATCH_SIZE,
        replacement=True,
    )
    test_sampler = WeightedRandomSampler(
        weights=[
            test_deny_weight if l > 0 else test_approve_weight for l in test_labels_array
        ],
        num_samples=BATCH_SIZE,
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_sampler=[train_sampler for _ in range(SAMPLES_PER_EPOCH)]
    )
    test_loader = DataLoader(
        test_dataset, batch_sampler=[test_sampler for _ in range(SAMPLES_PER_EPOCH)]
    )

    # Defining the LSTM


    class LSTMBinaryClassifier(nn.Module):
        def __init__(
            self,
            vocab,
            embedding_dim,
            hidden_dim,
            output_dim,
            n_layers,
            bidirectional,
            dropout,
        ):
            super().__init__()
            self.vocab = vocab
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim
            self.bidirectional = bidirectional
            self.embedding = nn.Embedding(len(vocab), embedding_dim)
            self.lstm = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, text, text_lengths):
            embedded = self.dropout(self.embedding(text))
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths.cpu(), batch_first=True
            )
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            last_hidden = hidden[-1, :, :]
            last_hidden = self.dropout(last_hidden)
            output = self.fc(last_hidden)
            return output

        def init_hidden(self, batch_size):
            if self.bidirectional:
                bidirectional_multiplicand = 2
            else:
                bidirectional_multiplicand = 1
            h0 = torch.zeros(
                (self.n_layers * bidirectional_multiplicand, batch_size, self.hidden_dim),
                dtype=torch.float,
            ).to(device)
            c0 = torch.zeros(
                (self.n_layers * bidirectional_multiplicand, batch_size, self.hidden_dim),
                dtype=torch.float,
            ).to(device)
            hidden = (h0, c0)
            return hidden


    criterion = nn.BCEWithLogitsLoss(
        reduction="sum"
    )  # pos_weight=torch.tensor([(labels==0).sum()/labels.sum()]))
    criterion.to(device)
    last_acc_total_loss = 999999999999999



    # Initialize the model
    model = LSTMBinaryClassifier(
        vocab=vocab,
        embedding_dim=1000,
        hidden_dim=256,
        output_dim=1,
        n_layers=2,
        bidirectional=True,
        dropout=DROPOUT_RATE,
    )

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=True)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        hidden_cell_tuple = model.init_hidden(BATCH_SIZE)
        losses = []
        for batch_num, batch in enumerate(train_loader):
            model.zero_grad()
            targets = batch["label"].int()
            hidden_cell_tuple = tuple([each.data for each in hidden_cell_tuple])
            text_split_idx = [
                torch.tensor(vocab.lookup_indices(text.split()))
                for text in batch["text"]
            ]
            text_split_idx_padded = nn.utils.rnn.pad_sequence(
                text_split_idx, batch_first=True
            ).int()
            text_split_idx_padded = text_split_idx_padded.to(device)
            text_lengths = torch.tensor(
                [len(text) for text in text_split_idx_padded], dtype=torch.int32
            )
            text_lengths = text_lengths.to(device)
            predictions = model(text_split_idx_padded, text_lengths)
            predictions = predictions.squeeze(1)
            # Prints out preditions, targets, and accuracy for the current batch
            print(
                "P",
                torch.round(torch.sigmoid(predictions.cpu())).int(),
                "\tBatch",
                batch_num + 1,
                "Epoch",
                epoch + 1,
            )
            print("T", targets, "\tBatch", batch_num + 1, "Epoch", epoch + 1)
            print(
                "A",
                torch.eq(torch.round(torch.sigmoid(predictions.cpu())).int(), targets)
                .float()
                .mean(),
                "\tBatch",
                batch_num + 1,
                "Epoch",
                epoch + 1,
            )
            predictions = predictions.to(device)
            targets = targets.to(device)
            loss = criterion(predictions.float(), targets.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        # Validation loop
        print("Starting testing for Epoch", epoch + 1)

        model.eval()
        total_loss = 0
        test_y_true = []
        test_y_pred = []
        hidden_cell_tuple = model.init_hidden(BATCH_SIZE)
        for batch in test_loader:
            targets = batch["label"].int()
            text_split_idx = [
                torch.tensor(vocab.lookup_indices(text.split()))
                for text in batch["text"]
            ]
            text_split_idx_padded = nn.utils.rnn.pad_sequence(
                text_split_idx, batch_first=True
            ).int()
            text_split_idx_padded = text_split_idx_padded.to(device)
            text_lengths = torch.tensor(
                [len(text) for text in text_split_idx_padded], dtype=torch.int32
            )
            text_lengths = text_lengths.to(device)
            with torch.no_grad():
                predictions = model(text_split_idx_padded, text_lengths)
                predictions = predictions.squeeze(1)
                predictions = predictions.to(device)
                targets = targets.to(device)
                loss = criterion(predictions.float(), targets.float())
            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(predictions))
            for p in preds:
                test_y_pred.append(p.cpu().item())
            for t in targets:
                test_y_true.append(t.cpu().item())

        print(f"Epoch: {epoch+1}, Testing Loss: {total_loss}")
        print(
            "accuracy",
            "precision",
            "recall",
            "f1",
            f"f{FBETA}",
        )
        print(
            [
                round(m, 5)
                for m in [
                    accuracy_score(test_y_true, test_y_pred),
                    precision_score(test_y_true, test_y_pred),
                    recall_score(test_y_true, test_y_pred),
                    f1_score(test_y_true, test_y_pred),
                    fbeta_score(test_y_true, test_y_pred, beta=FBETA),
                ]
            ],
        )
        if accuracy_score(test_y_true, test_y_pred) > last_acc_total_loss:
            pass# break
        last_acc_total_loss = accuracy_score(test_y_true, test_y_pred)

    torch.save(model, f"LSTM-{str(LEARNING_RATE).replace(".","")}-{str(DROPOUT_RATE).replace(".","")}")
else:
    torch.load(LOAD_MODEL)

eval_approve_texts, eval_approve_labels = load_text_files(r'data/finding_of_fact/eval/approve')
eval_deny_texts, eval_deny_labels = load_text_files(r'data/finding_of_fact/eval/deny')
eval_texts = eval_approve_texts + eval_deny_texts
eval_labels = np.array(list(map(lambda x: 0 if x.endswith("approve") else 1,eval_approve_labels + eval_deny_labels)))
eval_dataset = Dataset.from_dict({'text': eval_texts, 'label': eval_labels})
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE,shuffle=True, drop_last=True)

total_eval_loss = 0
eval_y_true = []
eval_y_pred = []
eval_hidden = model.init_hidden(BATCH_SIZE)
for batch in eval_loader:
    eval_text_split_idx = [torch.tensor(vocab.lookup_indices(text.split())) for text in batch["text"]]
    eval_text_split_idx_padded = nn.utils.rnn.pad_sequence(eval_text_split_idx,batch_first=True).int()
    eval_text_split_idx_padded = eval_text_split_idx_padded.to(device)
    eval_text_lengths = torch.tensor([len(text) for text in eval_text_split_idx_padded],dtype=torch.int32)
    eval_text_lengths = eval_text_lengths.to(device)

    with torch.no_grad():
        eval_predictions= model(eval_text_split_idx_padded, eval_text_lengths)
        eval_predictions = eval_predictions.squeeze(1)
        eval_targets = batch["label"].float()
        eval_targets = eval_targets.to(device)
        eval_loss = criterion(eval_predictions, eval_targets)
        total_eval_loss += eval_loss.item()
        preds = torch.round(torch.sigmoid(eval_predictions))
        for p in preds:
            eval_y_pred.append(p.cpu().item())
        for t in eval_targets:
            eval_y_true.append(t.cpu().item())

print(f"Average evaluation Loss: {total_eval_loss/len(eval_loader)}")
print("accuracy","precision","recall","f1",f"f{FBETA}",step="\t")
print(
    accuracy_score(eval_y_true,eval_y_pred), 
    precision_score(eval_y_true,eval_y_pred), 
    recall_score(eval_y_true,eval_y_pred), 
    f1_score(eval_y_true,eval_y_pred),
    fbeta_score(eval_y_true,eval_y_pred,beta=FBETA),
    sep="\t"
)
print("Displaying Confusion Matrix")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay(confusion_matrix(eval_y_true,eval_y_pred),display_labels=["approve","deny"]).plot()
plt.show()