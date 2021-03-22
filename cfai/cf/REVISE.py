#Filename:	REVISE.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Jum 01 Jan 2021 02:05:26  WIB

import torch
import numpy as np
import torch.nn.functional as F
import copy
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from cf.baseCF import ExplainerBase

class REVISE(ExplainerBase):

    def __init__(self, data_interface, model_interface, model_vae):
        super().__init__(data_interface, model_interface)
        self.model_vae = model_vae
        self.model_vae.eval()

    def generate_counterfactuals(self, query_instance, features_to_vary, target = 0.7, feature_weights = None,
            _lambda = 0.001, optimizer = "adam", lr = 3, max_iter = 300):

        start_time = time.time()
        if isinstance(query_instance, dict) or isinstance(query_instance, list):
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
        
        with torch.no_grad():
            mu, log_var = self.model_vae.encode(cf_initialize)
            z = self.model_vae.reparameterize(mu, log_var)
            cf = self.model_vae.decode(z)

        if optimizer == "adam":
            optim = torch.optim.Adam([cf], lr)
        else:
            optim = torch.optim.RMSprop([cf], lr)

        for i in range(max_iter):

            cf.requires_grad = True
            optim.zero_grad()
            #cf = self.model_vae.decode(z)
            loss = self.compute_loss(cf, query_instance, target)
            loss.backward()
            optim.step()
            cf.detach_()
            
        end_time = time.time()
        running_time = time.time()
        final_cf = self.model_vae.decode(z)

        return final_cf.numpy()

       
    def compute_loss(self, cf_initialize, query_instance, target):

        loss1 = F.relu(target - self.model_interface.predict_tensor(cf_initialize)[1])
        loss2 = torch.sum((cf_initialize - query_instance)**2)
        print(loss1, "\t", loss2)
        return loss1 + self._lambda * loss2

