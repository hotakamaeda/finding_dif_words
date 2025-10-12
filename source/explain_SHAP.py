

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shap
# import gc
from transformers import TrainingArguments, Trainer

# ----------------------------------------------------------------------------------------------------------------------
# Settings

def SHAP_3cat(input_name = 'female_3catd1',
              DATA_TYPE = 'data_type',
              LABEL = 'lab_female_3',
              out_label = '',
              only_testing = False,
              Fixed_context=1
              ):

    # ----------------------------------------------------------------------------------------------------------------------
    # Load model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    # max_length is set later.
    # max_length = 512  # this is important. High values like 700 do not work.
    # max_length = 400  # this is important. High values like 700 do not work.

    # ----------------------------------------------------------------------------------------------------------------------
    # Load data (we just need the text column)
    items = pd.read_csv("../output/baseline.csv", low_memory=False)

    # Remove rows based on Label
    # Keep only limited columns
    items = items[items[LABEL + 'ref'].notna()][["unique_id", "data_type", "data_type2", "text", "tokens"]]

    # ----------------------------------------------------------------------------------------------------------------------
    # Load fine-tuned model from local folder
    #  send model to GPU. GPU is much faster than CPU
    model = AutoModelForSequenceClassification.from_pretrained("../output/model_3j_" + input_name).to("cuda")

    # define a prediction function
    def f(x):
        encodings = [tokenizer.encode(v, padding='max_length', max_length=max_length, truncation=True) for v in x]
        tv = torch.tensor(encodings).cuda()
        outputs = model(tv)
        predictions = outputs[0]
        out = torch.nn.functional.softmax(predictions, dim=1)
        return out

    # ----------------------------------------------------------------------------------------------------------------------
    # Explainer

    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer, algorithm="partition")

    explainer_list = []
    count = 0
    for k, v in items.iterrows():
        count = count + 1
        # break
        if (only_testing & (v[DATA_TYPE] != "testing")):
            continue
        max_length = min([v.tokens, 512])
        if Fixed_context==None:
            attributions = explainer([v["text"]], fixed_context=None, batch_size=8)
        else:
            batch_size = int(np.floor(8*512/v.tokens))
            attributions = explainer([v["text"]], fixed_context=1, batch_size = batch_size)

        tokens = np.array(attributions.data).tolist()[0]
        values = attributions.values[0].tolist()
        d = {'unique_id': np.repeat(v["unique_id"], len(tokens)),
             'order': np.arange(0, len(tokens)),
             'token': tokens,
             'attribution0': [i[0] for i in values],
             'attribution1': [i[1] for i in values],
             'attribution2': [i[2] for i in values]
             }
        explainer_list.append(pd.DataFrame(d))
        print(input_name + out_label, count, v.tokens, end='\r')
    explainer_out = pd.concat(explainer_list)

    # # ----------------------------------------------------------------------------------------------------------------------
    # Save
    explainer_out.to_csv(path_or_buf="../output/SHAP_" + input_name + out_label + ".csv", index=False)

    print(input_name + out_label + ' done')

    return input_name + out_label + ' done'










