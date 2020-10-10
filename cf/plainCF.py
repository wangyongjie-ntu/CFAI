#Filename:	plainCF.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Jum 09 Okt 2020 12:12:31  WIB
#Paper:         Counterfactual Explanations without Opening the Black Box Automated Decisions and the GDPR
#Objective:     argmin_{x'}max_{\lambda} \lambda (f(x') - y')^2 + d(x_i, x')

import torch
import numpy as np
import random
import copy
import time

from cf.explainer_base import ExplainerBase

class PlainCF(ExplainerBase):

    def __init__(self, data_interface, model_interface, _lambda):

        """
        Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        super().__init__(data_interface)

        self.model = model_interface
        self.model.load_model()
        self.model.set_eval_mode()
        self._lambda = _lambda
        self.minx = torch.FloatTensor(self.data_interface.minx)
        self.maxx = torch.FloatTensor(self.data_interface.maxx)

    def generate_counterfactuals(self, query_instance, desired_target, features_to_vary, feature_weights = None, optimizer = "adam", lr = 0.01, max_iter = 1000, dist_type = "l2", loss_diff_thres = 0):

        """Generates diverse counterfactual explanations
        
        :param query_instance: the instance to be explained
        :param desired_target: Desired target compared with the current predicted target. Desired_target = test_pred * 1.1
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as values. Default option is "inverse_mad" where the weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set; the weight for a categorical feature is equal to 1 by default.
        :param optimizer: PyTorch optimization algorithm. Currently tested only with "pytorch:adam".
        :param lr: Learning rate for optimizer.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence. Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).

        """
    
        start_time = time.time()
        # The mask of feature to vary
        self.features_to_vary = features_to_vary
        self.features_to_vary_mask = torch.FloatTensor(self.data_interface.get_mask_of_features_to_vary(features_to_vary))

        if feature_weights == None:
            feature_weights = np.ones(len(self.data_interface.all_feature_names))
            self.feature_weights = torch.FloatTensor(feature_weights)
        else:
            feature_weights = self.data_interface.get_weights_of_features_to_vary(feature_weights)
            self.feature_weights = torch.FloatTensor(feature_weights)

        self.dist_type = dist_type
        
        query_instance = self.data_interface.normalize_data(query_instance)
        query_instance = torch.FloatTensor(query_instance)

        # find the predicted value of query_instance [1 * N]
        test_pred = self.predict_fn(query_instance).detach().numpy()
        raw_target = self.data_interface.de_normalize_target(test_pred)
        tmp = raw_target * desired_target
        target = self.data_interface.normalize_target(tmp)
        target = torch.FloatTensor(target)
        
        # initialize from the uniform distribution, except the immutable features
        cf_instance = np.random.uniform(self.data_interface.minx, self.data_interface.maxx)
        cf_instance = torch.FloatTensor(cf_instance)
        cf_instance = cf_instance * self.features_to_vary_mask + (1 - self.features_to_vary_mask) * query_instance
        
        if optimizer == "adam":
            optim = torch.optim.Adam([cf_instance], lr)
        else:
            optim = torch.optim.RMSprop([cf_instance], lr)

        for i in range(max_iter):
            
            cf_instance.requires_grad = True
            optim.zero_grad()
            loss = self.compute_loss(cf_instance, query_instance, target)
            loss.backward()
            cf_instance.grad = cf_instance.grad * self.features_to_vary_mask
            optim.step()
            
            # element-wise clamp
            cf_instance = torch.where(cf_instance > self.maxx, self.maxx, cf_instance)
            cf_instance = torch.where(cf_instance < self.minx, self.minx, cf_instance)

            cf_instance.detach_()

        end_time = time.time()
        running_time = end_time - start_time

        results = self.data_interface.de_normalize_data(cf_instance.numpy())
        
        return results

    def predict_fn(self, input_instance):
        """prediction function"""
        if not torch.is_tensor(input_instance):
            input_instance = torch.FloatTensor(input_instance)

        return self.model.get_output(input_instance)

    def compute_loss(self, cf_instance, query_instance, target):

        loss1 = (self.predict_fn(cf_instance) - target)**2
        loss2 = self.weighted_dist(cf_instance, query_instance)

        return self._lambda * loss1 + loss2

    def weighted_dist(self, cf_instance, query_instance):

        if self.dist_type == "inverse_mad":
            diff = torch.abs(cf_instance - query_instance)
            diff = diff * self.feature_to_vary_mask
            return torch.sum(diff / self.data_interface.mads)

        elif self.dist_type == "inverse_std":
            diff = torch.abs(cf_instance - query_instance)**2
            diff = diff * self.feature_to_vary_mask
            return torch.sum(diff / self.data_interface.stds)

        else:
            # default is l2 distance
            return torch.sum(torch.abs(cf_instance - query_instance)**2 * self.features_to_vary_mask)


