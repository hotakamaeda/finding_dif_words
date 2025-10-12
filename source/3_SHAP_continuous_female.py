
# ----------------------------------------------------------------------------------------------------------------------
# Explain the Continuous female/male DIF prediction model with SHAP

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shap
# import gc
from transformers import TrainingArguments, Trainer

# ----------------------------------------------------------------------------------------------------------------------
# Settings

LABEL = 'lab_female'
DATA_TYPE = 'data_type'
out_label = 'female_512reg1'

# ----------------------------------------------------------------------------------------------------------------------
# Load model, tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
# max_length = 512  # this is important. High values like 700 do not work.
# max_length = 400  # this is important. High values like 700 do not work.

# ----------------------------------------------------------------------------------------------------------------------
# Load data (we just need the text column)
items = pd.read_csv("../output/predicted_3j_" + out_label + ".csv", low_memory=False)

# ----------------------------------------------------------------------------------------------------------------------
# Load fine-tuned model from local folder
#  send model to GPU. GPU is much faster than CPU but can only handle about 100 tokens or less.
model = AutoModelForSequenceClassification.from_pretrained("../output/model_3j_" + out_label).to("cuda")

# define a prediction function
def f(x):
    encodings = [tokenizer.encode(v, padding='max_length', max_length=max_length, truncation=True) for v in x]
    tv = torch.tensor(encodings).cuda()
    outputs = model(tv)
    logits = outputs[0][:, 0]
    return logits

# ----------------------------------------------------------------------------------------------------------------------
# Explainer

# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer, seed=300)

explainer_list = []
count = 0
for k, v in items.iterrows():
    count = count + 1
    # break
    if v[DATA_TYPE] != "testing":
        continue
    batch_size = int(np.floor(8*512/v.tokens))
    max_length = min([v.tokens, 512])
    attributions = explainer([v["text"]], fixed_context=1, batch_size = batch_size)
    tokens = np.array(attributions.data).tolist()[0]
    values = attributions.values[0].tolist()
    d = {'unique_id': np.repeat(v["unique_id"], len(tokens)),
         'order': np.arange(0, len(tokens)),
         'token': tokens,
         'attribution': values
         }
    explainer_list.append(pd.DataFrame(d))
    print(count, v.tokens)
explainer_out = pd.concat(explainer_list)

# # ----------------------------------------------------------------------------------------------------------------------
# Save
explainer_out.to_csv(path_or_buf="../output/SHAP_" + out_label + ".csv", index=False)











