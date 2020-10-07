#Filename:	test_mice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 05 Okt 2020 03:34:23  WIB

from data.data import Data
from cf.dice_pytorch import DicePyTorch
from model.model import Model
from utils import helpers

import torch
import pandas as pd

if __name__ == "__main__":

    dataset = helpers.load_adult_income_dataset()

    #print(dataset.head())

    adult_info = helpers.get_adult_data_info()
    d = Data(dataframe=dataset, continuous_features=['age', 'hours_per_week'], outcome_name='income')
    m = Model(model_path= "./weigths/adult.pth")
    exp = DicePyTorch(d, m)

    query_instance = {'age':22,
                  'workclass':'Private',
                  'education':'HS-grad',
                  'marital_status':'Single',
                  'occupation':'Service',
                  'race': 'White',
                  'gender':'Female',
                  'hours_per_week': 45}

    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_class="opposite", features_to_vary = ["education", "marital_status", "occupation", "hours_per_week"])
    
    print("original instance")
    print(dice_exp.org_instance)

    print("counterfactual instances found")
    print(dice_exp.final_cfs_list)


