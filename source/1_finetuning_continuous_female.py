
# ----------------------------------------------------------------------------------------------------------------------
# Fine-tune deberta to predict continuous female/male DIF

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import copy
from transformers import TrainingArguments, Trainer
import torch
import re
from transformers import DataCollatorWithPadding
from sklearn.metrics import mean_squared_error, r2_score

torch.cuda.is_available()


# ----------------------------------------------------------------------------------------------------------------------
# Settings

max_length = 512
LABEL = 'lab_female'
DATA_TYPE = 'data_type'
out_label = 'female_512reg1'
np.random.seed(seed=1)

# ----------------------------------------------------------------------------------------------------------------------
# Functions

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0][:, 0]
        loss = torch.nn.functional.mse_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)
    mse = round(mean_squared_error(labels, logits), 4)
    r2 = round(r2_score(labels, logits), 4)
    return {"mse": mse, "r2": r2}


# ----------------------------------------------------------------------------------------------------------------------
# Load model, tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
model0 = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large",
    num_labels=1
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------------------------------------------------------------------------------------------------
# Load data
items = pd.read_csv("../output/baseline.csv", low_memory=False)
items = items[(items[LABEL].notna()) & (items[LABEL] != "")]

# Text and label
train_texts = items.loc[items[DATA_TYPE] == "training", 'text'].tolist()
eval_texts = items.loc[items[DATA_TYPE] == "evaluation", 'text'].tolist()
train_labels = items.loc[items[DATA_TYPE] == "training", LABEL].tolist()
eval_labels = items.loc[items[DATA_TYPE] == "evaluation", LABEL].tolist()
print(np.var(train_labels))
print(np.var(eval_labels))

# Tokenize. Truncate at max length
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=max_length)

# Dataset
train_dataset = Dataset(train_encodings, [float(i) for i in train_labels])
eval_dataset = Dataset(eval_encodings, [float(i) for i in eval_labels])

# ----------------------------------------------------------------------------------------------------------------------
# FINE-TUNE
# Load base model (reset model)
model = copy.deepcopy(model0).to("cuda") # copy cpu default model, send to GPU
torch.device("cuda")
# ff.print_gpu_utilization()

training_args = TrainingArguments(
    output_dir="../output/saved_models",
    optim="adamw_torch",
    learning_rate=4e-6,
    per_device_train_batch_size=8,  # lower batch size if memory runs out.
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True, # this loads the best model out of all epochs. So every epoch must be saved
    save_total_limit=2,
    fp16=True,  # use if memory runs out. This uses only 4 decimal points
)

trainer = RegressionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_for_regression,
)

trainer.train()
trainer.save_model("../output/model_3j_" + out_label)
LOSS = pd.DataFrame(trainer.state.log_history)

 # ------------------------------------------------------------------------------------------------------------------
# PREDICT WITH ALL DATA

# Text and label
text2 = items["text"].tolist()
label = items[LABEL].tolist()
# Tokenize. Truncate at max length
encodings = tokenizer(text2, truncation=True, padding=True, max_length=max_length)
dataset = Dataset(encodings, [float(i) for i in label])

# Predict
predicted = trainer.predict(dataset)
items['pred'] = predicted.predictions

# ----------------------------------------------------------------------------------------------------------------------
# SAVE DATA
items.to_csv(path_or_buf="../output/predicted_3j_" + out_label + ".csv", index=False)
LOSS.to_csv(path_or_buf="../output/LOSS_3j_" + out_label + ".csv", index=False)






















