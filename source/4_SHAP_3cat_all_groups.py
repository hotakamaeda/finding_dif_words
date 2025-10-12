
# ----------------------------------------------------------------------------------------------------------------------
# This is a fine-tuning run predicting DIF stats with deberta regression

import torch
import pandas as pd
import numpy as np
import source.explain_SHAP as e

print(torch.cuda.is_available())

# ----------------------------------------------------------------------------------------------------------------------
# Explain

# Female (run continuous regression and logit versions separately)
e.SHAP_3cat(input_name = 'female_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_female_3',
            out_label = '',
            only_testing = True)

# Other groups
e.SHAP_3cat(input_name = 'asian_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_asian_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'black_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_black_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'hispanic_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_hispanic_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'native_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_native_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'idea_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_idea_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'lep_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_lep_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'econ_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_econ_3',
            out_label = '',
            only_testing = True)

###################################################################################################
# Different Seed

e.SHAP_3cat(input_name = 'female_3catd1seed',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_female_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'asian_3catd1seed',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_asian_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'black_3catd1seed',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_black_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'hispanic_3catd1seed',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_hispanic_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'native_3catd1seed',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_native_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'idea_3catd1seed',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_idea_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'lep_3catd1seed',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_lep_3',
            out_label = '',
            only_testing = True)

e.SHAP_3cat(input_name = 'econ_3catd1seed',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_econ_3',
            out_label = '',
            only_testing = True)



# Fixed context?
e.SHAP_3cat(input_name = 'female_3catd1',
            DATA_TYPE = 'data_type',
            LABEL = 'lab_female_3',
            out_label = '_fixed_context',
            only_testing = True,
            Fixed_context=None)




