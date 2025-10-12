
# ----------------------------------------------------------------------------------------------------------------------
# This is a fine-tuning run predicting DIF stats with deberta regression

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import numpy as np
# import copy
from transformers import TrainingArguments, Trainer
import torch
import re
from transformers import DataCollatorWithPadding
import random
import datasets
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------------------------------------------------
# Replacement with 3 category model

def pred_rep_cat3model(
        input_name = 'female_3catd1',
        out_label = "",
        LABEL = 'lab_female_3',
        MAX=True):

    max_length = 512
    DATA_TYPE = 'data_type'

    # ----------------------------------------------------------------------------------------------------------------------
    # Load model, tokenizer, SHAP results
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained("../output/model_3j_" + input_name).to("cuda")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    SHAP = pd.read_csv("../output/SHAP_" + input_name + ".csv", low_memory=False)
    SHAP.loc[SHAP.attribution0 < 0, "attribution0"] = 0
    SHAP.loc[SHAP.attribution2 < 0, "attribution2"] = 0
    SHAP["shap"] = -SHAP.attribution0 + SHAP.attribution2
    # SHAP['abs'] = abs(SHAP["shap"])

    # ----------------------------------------------------------------------------------------------------------------------
    # Load data (we just need the text column)
    items = pd.read_csv("../output/baseline.csv", low_memory=False)

    # Remove rows based on Label
    # Keep only limited columns
    LABEL0 = LABEL + 'ref'
    items = items[items[LABEL + 'ref'].notna()][["unique_id", "data_type", "data_type2", "text", "tokens", LABEL0]]

    # Only test data
    items = items[items[DATA_TYPE] == "testing"]

    # ----------------------------------------------------------------------------------------------------------------------
    # Prepare Label and Text Data
    # Text and label
    test_texts = items['text'].tolist()

    # ----------------------------------------------------------------------------------------------------------------------
    # Predict

    OUT = []
    count = int(-1)
    with torch.no_grad():
        test_encodings = tokenizer(test_texts, padding=False, max_length=max_length)
        for id in items.unique_id:
            count = int(count + 1)
            # Find most extreme SHAP value
            SHAP1 = SHAP[SHAP.unique_id == id]
            if MAX:
                index_max = max(SHAP1["order"].tolist(), key=SHAP1["shap"].tolist().__getitem__)
            else:
                index_max = min(SHAP1["order"].tolist(), key=SHAP1["shap"].tolist().__getitem__)
            input_ids = test_encodings["input_ids"][count]
            # 3 is [UNK]  (unknown token)
            input_ids[index_max] = 3
            # break
            out = model(torch.tensor([input_ids]).to("cuda"))
            logits = out.logits.cpu()
            EXP = np.exp(logits.tolist()[0])
            preds = EXP/sum(EXP)
            d = {'unique_id': [id],
                 'pred0': [preds[0]],
                 'pred1': [preds[1]],
                 'pred2': [preds[2]]
                 }
            OUT.append(pd.DataFrame(d))
            print(input_name, out_label, count, end='\r')
    OUT1 = pd.concat(OUT)

    # ----------------------------------------------------------------------------------------------------------------------
    # SAVE DATA
    OUT1.to_csv(path_or_buf="../output/predicted_replaced_" + input_name + out_label + ".csv", index=False)

    print(input_name + out_label + ' done')

    return input_name + out_label + ' done'



# ----------------------------------------------------------------------------------------------------------------------
# Replacement with continuous model

def pred_rep_contmodel(
    input_name = 'female_512reg1',
    out_label = '',
    LABEL = 'lab_female',
    MAX=True
    ):

    max_length = 512
    DATA_TYPE = 'data_type'

    # ----------------------------------------------------------------------------------------------------------------------
    # Load model, tokenizer, SHAP results
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    model = AutoModelForSequenceClassification.from_pretrained("../output/model_3j_" + input_name).to("cuda")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    SHAP = pd.read_csv("../output/SHAP_" + input_name + ".csv", low_memory=False)
    # SHAP['abs'] = abs(SHAP["attribution"])

    # ----------------------------------------------------------------------------------------------------------------------
    # Load data (we just need the text column)
    items = pd.read_csv("../output/baseline.csv", low_memory=False)

    # Remove rows based on Label
    # Keep only limited columns
    items = items[items[LABEL].notna()][["unique_id", "data_type", "data_type2", "text", "tokens", LABEL]]

    # Only test data
    items = items[items[DATA_TYPE] == "testing"]

    # ----------------------------------------------------------------------------------------------------------------------
    # Prepare Label and Text Data
    # Text and label
    test_texts = items['text'].tolist()

    # ----------------------------------------------------------------------------------------------------------------------
    # Predict

    OUT = []
    count = int(-1)
    with torch.no_grad():
        test_encodings = tokenizer(test_texts, padding=False, max_length=max_length)
        for id in items.unique_id:
            count = int(count + 1)
            # break
            # Find most extreme SHAP value
            SHAP1 = SHAP[SHAP.unique_id == id]
            if MAX:
                index_max = max(SHAP1["order"].tolist(), key=SHAP1["attribution"].tolist().__getitem__)
            else:
                index_max = min(SHAP1["order"].tolist(), key=SHAP1["attribution"].tolist().__getitem__)
            input_ids = test_encodings["input_ids"][count]
            # 3 is [UNK]  (unknown token)
            input_ids[index_max] = 3
            # break
            out = model(torch.tensor([input_ids]).to("cuda"))
            logits = out.logits.cpu().tolist()
            d = {'unique_id': [id],
                 'pred0': logits[0]
                 }
            OUT.append(pd.DataFrame(d))
            print(input_name, out_label, count, end='\r')
    OUT1 = pd.concat(OUT)

    # ----------------------------------------------------------------------------------------------------------------------
    # SAVE DATA
    OUT1.to_csv(path_or_buf="../output/predicted_replaced_" + input_name + out_label + ".csv", index=False)

    print(input_name + out_label + ' done')

    return input_name + out_label + ' done'

