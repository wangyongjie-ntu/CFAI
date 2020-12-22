#Filename:	plainCF.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 13 Des 2020 09:15:05  WIB

import torch
import numpy as np
import torch.nn.functional as F
import copy
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from cf.baseCF import ExplainerBase

class PlainCF(ExplainerBase):

    def __init__(self, data_interface, model_interface):

        super().__init__(data_interface, model_interface)

    def generate_counterfactuals(self, query_instance, features_to_vary, target = 0.7, feature_weights = None, _lambda = 10,
            optimizer = "adam", lr = 0.01, max_iter = 100):
       
        start_time = time.time()
        query_instance = self.data_interface.prepare_query(query_instance, normalized = True)
        query_instance = torch.FloatTensor(query_instance)

        mask = self.data_interface.get_mask_of_features_to_vary(features_to_vary)
        mask = torch.LongTensor(mask)

        self._lambda = _lambda

        if feature_weights == None:
            feature_weights = torch.ones(query_instance.shape[1])
        else:
            feature_weights = torch.ones(query_instance.shape[0])
            feature_weights = torch.FloatTensor(feature_weights)

        if isinstance(self.data_interface.scaler, MinMaxScaler):
            cf_initialize = torch.rand(query_instance.shape)
        elif isinstance(self.data_interface.scaler, StandardScaler):
            cf_initialize = torch.randn(query_instance.shape)
        else:
            cf_initialize = torch.rand(query_instance.shape)

        cf_initialize = torch.FloatTensor(cf_initialize)
        cf_initialize = mask * cf_initialize + (1 - mask) * query_instance
        
        if optimizer == "adam":
            optim = torch.optim.Adam([cf_initialize], lr)
        else:
            optim = torch.optim.RMSprop([cf_initialize], lr)

        for i in range(max_iter):
            cf_initialize.requires_grad = True
            optim.zero_grad()
            loss = self.compute_loss(cf_initialize, query_instance, target)
            loss.backward()
            cf_initialize.grad = cf_initialize.grad * mask
            optim.step()
            
            if isinstance(self.data_interface.scaler, MinMaxScaler):
                cf_initialize = torch.where(cf_initialize > 1, torch.ones_like(cf_initialize), cf_initialize)
                cf_initialize = torch.where(cf_initialize < 0, torch.zeros_like(cf_initialize), cf_initialize)

            cf_initialize.detach_()

        end_time = time.time()
        running_time = time.time()

        return cf_initialize

    def compute_loss(self, cf_initialize, query_instance, target):

        loss1 = F.relu(target - self.model_interface.predict_tensor(cf_initialize)[1])
        loss2 = torch.sum((cf_initialize - query_instance)**2)
        return self._lambda * loss1 + loss2

