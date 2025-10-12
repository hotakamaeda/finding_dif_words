
# ----------------------------------------------------------------------------------------------------------------------
# Fine-tune deberta to predict 3-category DIF for 8 DIF groups

import torch
import pandas as pd
import numpy as np
import source.finetuning_3cat_model as c

print(torch.cuda.is_available())

# ----------------------------------------------------------------------------------------------------------------------
# Prediction

c.cat3model(out_label = 'female_3catd1',
            LABEL = 'lab_female_3',
            max_length = 512,
            DATA_TYPE = 'data_type',
            Full_data = False)

c.cat3model(out_label = 'asian_3catd1',
              LABEL = 'lab_asian_3',
              max_length = 512,
              DATA_TYPE = 'data_type')

c.cat3model(out_label = 'black_3catd1',
              LABEL = 'lab_black_3',
              max_length = 512,
              DATA_TYPE = 'data_type')

c.cat3model(out_label = 'hispanic_3catd1',
              LABEL = 'lab_hispanic_3',
              max_length = 512,
              DATA_TYPE = 'data_type')

c.cat3model(out_label = 'native_3catd1',
              LABEL = 'lab_native_3',
              max_length = 512,
              DATA_TYPE = 'data_type')

c.cat3model(out_label = 'idea_3catd1',
              LABEL = 'lab_idea_3',
              max_length = 512,
              DATA_TYPE = 'data_type')

c.cat3model(out_label = 'lep_3catd1',
              LABEL = 'lab_lep_3',
              max_length = 512,
              DATA_TYPE = 'data_type')

c.cat3model(out_label = 'econ_3catd1',
              LABEL = 'lab_econ_3',
              max_length = 512,
              DATA_TYPE = 'data_type')

# ----------------------------------------------------------------------------------------------------------------------
# Different seed

c.cat3model(out_label = 'female_3catd1seed',
              LABEL = 'lab_female_3',
              max_length = 512,
              DATA_TYPE = 'data_type',
            seed_ = 2)

c.cat3model(out_label = 'asian_3catd1seed',
            LABEL = 'lab_asian_3',
            max_length = 512,
            DATA_TYPE = 'data_type',
            seed_ = 2)

c.cat3model(out_label = 'black_3catd1seed',
              LABEL = 'lab_black_3',
              max_length = 512,
              DATA_TYPE = 'data_type',
            seed_ = 2)

c.cat3model(out_label = 'hispanic_3catd1seed',
              LABEL = 'lab_hispanic_3',
              max_length = 512,
              DATA_TYPE = 'data_type',
            seed_ = 2)

c.cat3model(out_label = 'native_3catd1seed',
              LABEL = 'lab_native_3',
              max_length = 512,
              DATA_TYPE = 'data_type',
            seed_ = 2)

c.cat3model(out_label = 'idea_3catd1seed',
              LABEL = 'lab_idea_3',
              max_length = 512,
              DATA_TYPE = 'data_type',
            seed_ = 2)

c.cat3model(out_label = 'lep_3catd1seed',
              LABEL = 'lab_lep_3',
              max_length = 512,
              DATA_TYPE = 'data_type',
            seed_ = 2)

c.cat3model(out_label = 'econ_3catd1seed',
              LABEL = 'lab_econ_3',
              max_length = 512,
              DATA_TYPE = 'data_type',
            seed_ = 2)




