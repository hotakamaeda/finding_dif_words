
# ----------------------------------------------------------------------------------------------------------------------
# This is a fine-tuning run predicting DIF stats with deberta regression

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
DATA_TYPE = 'data_type'

# ----------------------------------------------------------------------------------------------------------------------
# Load model, tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

# ----------------------------------------------------------------------------------------------------------------------
# Load data
items = pd.read_csv("../output/item_text_stats.csv", low_memory=False)

# ----------------------------------------------------------------------------------------------------------------------
# Prepare Label and Text Data

# replace None with ""
items.fillna("", inplace=True)

# Remove items with blank prompt
items = items[items['prompt'] != ""]

# Data text
# Move key to 1st option? [SEP] options
items['text'] = (
        # items['stim'] + '[SEP]' +
        # items['ela3stim'] + '[SEP]' +
        items['prompt'] + '[SEP]' +
        # items['stim'] + " " +
        # items['ela3stim'] + " " +
        # '[SEP]' +
        items['option1'] + '[SEP]' +
        items['option2'] + '[SEP]' +
        items['option3'] + '[SEP]' +
        items['option4'] + '[SEP]' +
        items['option5'] + '[SEP]' +
        items['option6'] + '[SEP]' +
        items['option7'] + '[SEP]' +
        items['option8'] + '[SEP]' +
        items['prompta'] + '[SEP]' +
        items['promptb'] + '[SEP]' +
        # + '[SEP]' +
        items['optiona1'] + '[SEP]' +
        items['optiona2'] + '[SEP]' +
        items['optiona3'] + '[SEP]' +
        items['optiona4'] + '[SEP]' +
        items['optionb1'] + '[SEP]' +
        items['optionb2'] + '[SEP]' +
        items['optionb3'] + '[SEP]' +
        items['optionb4'] + '[SEP]' +
        items['prompt_htqs'] + '[SEP]' +
        items['htqo1'] + '[SEP]' +
        items['htqo2'] + '[SEP]' +
        items['htqo3'] + '[SEP]' +
        items['htqo4'] + '[SEP]' +
        items['htqo5'] + '[SEP]' +
        items['htqo6'] + '[SEP]' +
        items['htqo7'] + '[SEP]' +
        items['htqo8'] + '[SEP]' +
        items['htqo9'] + '[SEP]' +
        items['htqo10'] + '[SEP]' +
        items['htqo11'] + '[SEP]' +
        items['htqo12'] + '[SEP]' +
        items['htqo13'] + '[SEP]' +
        items['htqo14'] + '[SEP]' +
        items['htqo15']
        # + '[SEP]'
        # items['stim'] + items['ela3stim']
        ).tolist()

# Remove Excess spaces, [SEP]
for k, v in items.iterrows():
    # Remove repeating spaces
    items.loc[k, 'text'] = re.sub(r'( )*\1', r'\1', items.loc[k, 'text'])
    # Remove repeating [SEP]
    items.loc[k, 'text'] = re.sub(r'(\[SEP\])*\1', r'\1', items.loc[k, 'text'])
    # Remove [SEP] from end of sentence
    items.loc[k, 'text'] = re.sub(r'\[SEP\]$', r'', items.loc[k, 'text'])

# Token count
items["tokens"] = 0
for k, v in items.iterrows():
    encodings = tokenizer(v["text"], truncation=True, padding=False, max_length=99999)
    items.loc[items["unique_id"] == v["unique_id"], "tokens"] = len(encodings["input_ids"])
# np.mean(items.tokens)
# np.mean(items.tokens < 400)

# ----------------------------------------------------------------------------------------------------------------------
# SAVE DATA
items.to_csv(path_or_buf="../output/baseline.csv", index=False)





