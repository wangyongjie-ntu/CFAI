#Filename:	test_mice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 05 Okt 2020 03:34:23  WIB

from data.data import Data
from cf.dice_pytorch import DicePyTorch
from model.model import Model
from utils import helpers

import torch
import json
import pandas as pd

if __name__ == "__main__":

    dataset = pd.read_csv("datasets/sample.csv")

    #print(dataset.head())

    m = Model(model_path= "../Supply-Chain/weights/1year_model_without_wlb.pth")
    d = Data(dataframe=dataset, categorical_features = [], target = 'diff_paid_order_num_td')
    exp = DicePyTorch(d, m)

    with open("datasets/sample.json", 'r') as f:
        query_instance = json.load(f)

    dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=4, desired_target = 1.1, features_to_vary = ["gmv_tn",  "coupon_fee_tn", "jhs_num_tn", "jhs_gmv_tn", "paid_order_num_last5m", "gmv_last5m"])
    
    print("original instance")
    print(dice_exp.org_instance)

    print("counterfactual instances found")
    print(dice_exp.final_cfs_list)


