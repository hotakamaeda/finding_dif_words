
# ----------------------------------------------------------------------------------------------------------------------
# See how the female/male DIF predictions change when highly positive and negative attribution words are replaced with [UNK]

import torch
import pandas as pd
import numpy as np
import source.prediction_w_replacement as p

print(torch.cuda.is_available())

# ----------------------------------------------------------------------------------------------------------------------
# Prediction with replacement of highly positive-attribution words with [UNK]

p.pred_rep_cat3model(
    input_name = 'female_3catd1',
    out_label = "_max",
    MAX=True,
    LABEL = 'lab_female_3')

p.pred_rep_cat3model(
    input_name = 'female_3catd1seed',
    out_label = "_max",
    MAX=True,
    LABEL = 'lab_female_3')

p.pred_rep_contmodel(
    input_name = 'female_512reg1',
    out_label = "_max",
    MAX=True,
    LABEL = 'lab_female')

p.pred_rep_contmodel(
    input_name = 'female_512reg1seed',
    out_label = "_max",
    MAX=True,
    LABEL = 'lab_female')

# ----------------------------------------------------------------------------------------------------------------------
# Prediction with replacement of highly-negative-attribution words with [UNK]

p.pred_rep_cat3model(
    input_name = 'female_3catd1',
    out_label = "_min",
    MAX=False,
    LABEL = 'lab_female_3')

p.pred_rep_cat3model(
    input_name = 'female_3catd1seed',
    out_label = "_min",
    MAX=False,
    LABEL = 'lab_female_3')

p.pred_rep_contmodel(
    input_name = 'female_512reg1',
    out_label = "_min",
    MAX=False,
    LABEL = 'lab_female')

p.pred_rep_contmodel(
    input_name = 'female_512reg1seed',
    out_label = "_min",
    MAX=False,
    LABEL = 'lab_female')

