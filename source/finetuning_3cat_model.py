
# ----------------------------------------------------------------------------------------------------------------------
# This is a fine-tuning run predicting DIF stats with deberta regression

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
# import copy
from transformers import TrainingArguments, Trainer
import torch
import re
from transformers import DataCollatorWithPadding
import random
import datasets

# ----------------------------------------------------------------------------------------------------------------------
# Functions

class Dataset_mult(torch.utils.data.Dataset):
    def __init__(self, encodings, labels0, labels1, labels2):
        self.encodings = encodings
        self.labels0 = labels0
        self.labels1 = labels1
        self.labels2 = labels2

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels0[idx], self.labels1[idx], self.labels2[idx]])
        return item

    def __len__(self):
        return len(self.labels0)


class RegressionTrainerMult_CEL(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        predictions = outputs[0]
        predictions = torch.nn.functional.softmax(predictions, dim=1)
        # print(predictions.size())
        # print(labels.size())
        CEL = -(labels * torch.log(predictions))
        loss = torch.mean(CEL)
        return (loss, outputs) if return_outputs else loss


def cat3model(out_label = 'asian_3catd1',
              LABEL = 'lab_asian_3',
              max_length = 512,
              DATA_TYPE = 'data_type',
              Full_data = False,
              seed_ = 1):

    # ----------------------------------------------------------------------------------------------------------------------
    # Random behavior

    # Python's built-in random module
    random.seed(seed_)

    # NumPy
    np.random.seed(seed_)

    # PyTorch (CPU and GPU)
    torch.manual_seed(seed_)
    torch.cuda.manual_seed(seed_)
    torch.cuda.manual_seed_all(seed_)  # If using multi-GPU setups

    # Ensure deterministic behavior in PyTorch (may affect performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------------------------------------------------------
    # Load model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-large",
        num_labels=3
        ).to("cuda")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # ----------------------------------------------------------------------------------------------------------------------
    # Load data
    items = pd.read_csv("../output/baseline.csv", low_memory=False)

    # ----------------------------------------------------------------------------------------------------------------------
    # Prepare Label and Text Data
    LABEL0 = LABEL + 'ref'
    LABEL1 = LABEL + 'neut'
    LABEL2 = LABEL + 'foc'

    # Remove rows based on Label
    items = items[(items[LABEL0].notna()) & (items[LABEL0] != "")]

    # Text and label
    train_texts = items.loc[items[DATA_TYPE] == "training", 'text'].tolist()
    eval_texts = items.loc[items[DATA_TYPE] == "evaluation", 'text'].tolist()
    train_labels0 = items.loc[items[DATA_TYPE] == "training", LABEL0].tolist()
    eval_labels0 = items.loc[items[DATA_TYPE] == "evaluation", LABEL0].tolist()
    train_labels1 = items.loc[items[DATA_TYPE] == "training", LABEL1].tolist()
    eval_labels1 = items.loc[items[DATA_TYPE] == "evaluation", LABEL1].tolist()
    train_labels2 = items.loc[items[DATA_TYPE] == "training", LABEL2].tolist()
    eval_labels2 = items.loc[items[DATA_TYPE] == "evaluation", LABEL2].tolist()

    # Tokenize. Truncate at max length
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    eval_encodings = tokenizer(eval_texts, truncation=True, padding=True, max_length=max_length)

    # Dataset
    train_dataset = Dataset_mult(train_encodings,
                                 [float(i) for i in train_labels0],
                                 [float(i) for i in train_labels1],
                                 [float(i) for i in train_labels2])
    eval_dataset = Dataset_mult(eval_encodings,
                                [float(i) for i in eval_labels0],
                                [float(i) for i in eval_labels1],
                                [float(i) for i in eval_labels2])

    # ----------------------------------------------------------------------------------------------------------------------
    # FINE-TUNE
    # Load base model (reset model)
    # model = copy.deepcopy(model0).to("cuda") # copy cpu default model, send to GPU
    torch.device("cuda")

    training_args = TrainingArguments(
        output_dir="../output/saved_models",
        optim="adamw_torch",
        learning_rate=4e-6,
        per_device_train_batch_size=8,  # lower batch size if memory runs out.
        per_device_eval_batch_size=8,
        num_train_epochs=2, # 2 was best
        weight_decay=0.001,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # this loads the best model out of all epochs. So every epoch must be saved
        save_total_limit=2,
        fp16=True,  # use if memory runs out. This uses only 4 decimal points
        disable_tqdm=True  # disable progress bar
    )

    trainer = RegressionTrainerMult_CEL(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("../output/model_3j_" + out_label)
    LOSS = pd.DataFrame(trainer.state.log_history)

     # ------------------------------------------------------------------------------------------------------------------
    # PREDICT WITH ALL DATA

    # Text and label
    text2 = items["text"].tolist()
    label0 = items[LABEL0].tolist()
    # Tokenize. Truncate at max length
    encodings = tokenizer(text2, truncation=True, padding=True, max_length=max_length)
    dataset = Dataset_mult(encodings,
                            # 3 values below does not matter, since we're predicting
                           [float(i) for i in label0],
                           [float(i) for i in label0],
                           [float(i) for i in label0])

    # Predict
    predicted = trainer.predict(dataset)
    # Softmax
    predicted0 = [i[0]/sum(i) for i in np.exp(predicted[0])]
    predicted2 = [i[2]/sum(i) for i in np.exp(predicted[0])]
    #
    items['pred0'] = predicted0
    items['pred2'] = predicted2
    items['pred1'] = 1 - items['pred0'] - items['pred2']

    if not Full_data:
        items = items[["unique_id", "pred0", "pred1", "pred2"]]
        # items = items[["unique_id", 'data_type', "data_type2", "text", "tokens",  "pred0", "pred1", "pred2"]]

    # ----------------------------------------------------------------------------------------------------------------------
    # SAVE DATA
    items.to_csv(path_or_buf="../output/predicted_3j_" + out_label + ".csv", index=False)
    LOSS.to_csv(path_or_buf="../output/LOSS_3j_" + out_label + ".csv", index=False)

    print(out_label + ' done')

    return out_label + ' done'





















