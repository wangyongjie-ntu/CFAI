#Filename:	GrowingSphere.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 17 Jan 2021 02:42:23  WIB

import torch
import numpy as np
import torch.nn.functional as F
import copy
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from cf.baseCF import ExplainerBase

class GrowingSphere(ExplainerBase):

    def __init__(self, data_interface, model_interface):

        super().__init__(data_interface, model_interface)

    def generate_counterfactuals(self, query_instance, features_to_vary, eta = 5, target = 0.7, observation_n = 20):

        start_time = time.time()
        if isinstance(query_instance, dict) or isinstance(query_instance, list):
            query_instance = self.data_interface.prepare_query(query_instance, normalized = True)

        mask = self.data_interface.get_mask_of_features_to_vary(features_to_vary)
        eta, zs = self.find_eta(eta, observation_n, query_instance.squeeze(), mask, target)
        enemy = self.find_enemy(eta, 2 * eta, zs, observation_n, query_instance.squeeze(), mask, target)
        
        if len(enemy.shape) == 0:
            enemy_star = self.find_sparse_enemy(enemy, target, query_instance.squeeze())
        else:
            enemy_star = np.zeros_like(enemy)
            for i in range(len(enemy)):
                enemy_star[i] = self.find_sparse_enemy(enemy[i], target, query_instance.squeeze())

        return enemy_star

    def check_if_in_SL(self, z, low, high, query_instance):

        norm_val = np.linalg.norm(query_instance - z)
        if norm_val >= low and norm_val <= high:
            return True
        else:
            return False

    def generate(self, eta, low, high, query_instance, mask):

        random_vector = np.random.uniform(-1, 1, len(query_instance)).astype(np.float32)
        random_vector = random_vector * mask
        a = eta / np.sqrt(np.sum(np.square(random_vector)))
        b = a * random_vector
        random_point = b + query_instance

        if (random_point > 1).any():
            random_point = np.minimum(random_point, 1)

        if (random_point < 0).any():
            random_point = np.maximum(random_point, 0)

        if self.check_if_in_SL(random_point, low, high, query_instance):
            return random_point
        else:
            return self.generate(eta, low, high, query_instance, mask)

    def make_z(self, eta, low, high, query_instance, mask, observation_n):
        zs = np.zeros((observation_n, len(query_instance))).astype(np.float32)
        for i in range(observation_n):
            zs[i] = self.generate(eta, low, high, query_instance, mask)

        return zs

    def binary_eta(self, z, eta, target):
        if (self.model_interface.predict_ndarray(z)[1] > target).any():
            return eta/2
        else:
            return None

    def find_eta(self, radius_eta, observation_n, query_instance, mask, target):
        eta = radius_eta
        zs = self.make_z(eta, 0, eta, query_instance, mask, observation_n)
        tmp = self.binary_eta(zs, eta, target)
        while tmp is not None:
            eta = tmp
            zs = self.make_z(tmp, 0, tmp, query_instance, mask, observation_n)
            tmp = self.binary_eta(zs, tmp, target)

        return eta, zs

    def find_enemy(self, a0, a1, zs, observation_n, query_instance, mask, target):

        eta = a0
        while True:
            if not (self.model_interface.predict_ndarray(zs)[1] > target).any():
                zs = self.make_z(a1, a0, a1, query_instance, mask, observation_n)
                a0 = a1
                a1 = a1 + eta
            else:
                break
        
        prediction = self.model_interface.predict_ndarray(zs)[1]
        idx = np.argwhere(prediction > target).squeeze()
        return zs[idx]

    def find_sparse_enemy(self, enemy, target, query_instance):

        enemy_prime = enemy.copy()
        non_zero_indices = np.argwhere(enemy != query_instance).squeeze().tolist()

        while self.model_interface.predict_ndarray(enemy_prime[np.newaxis, :])[1] > target:
            argmin = np.argmin(np.abs(enemy_prime[non_zero_indices] - query_instance[non_zero_indices]))
            enemy_prime[non_zero_indices[argmin]] = query_instance[non_zero_indices[argmin]]
            non_zero_indices.remove(non_zero_indices[argmin])
            
        return enemy_prime

