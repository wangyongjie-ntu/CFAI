#Filename:	test_mice.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 05 Okt 2020 03:34:23  WIB

from data.data import PublicData
from cf.plainCF import PlainCF
from model.model import Model

import torch
import json
import pandas as pd

if __name__ == "__main__":

    dataset = pd.read_csv("/home/yongjie/code/Supply-Chain/data/1month/1month_without_wlb_37.csv")

    #print(dataset.head())

    m = Model(model_path= "../Supply-Chain/weights/1month_model_without_wlb_37.pth")
    d = PublicData(dataframe=dataset, target = 'diff_paid_order_num_td')
    exp = PlainCF(d, m, _lambda = 100)

    query_instance = dataset.iloc[-1:, 0:-1].values

    dice_exp = exp.generate_counterfactuals(query_instance, desired_target = 1.1, features_to_vary = ['paid_order_num_t12', 'diff_price_sum', 'paid_order_num_m0', 'item_id_num', 'discount_avg', 'diff_price_avg'])
    
    print(query_instance)
    print(dice_exp)
    
    '''
    print("original instance")
    print(dice_exp.org_instance)

    print("counterfactual instances found")
    print(dice_exp.final_cfs_list)

    '''
