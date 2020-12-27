#Filename:	DiCE.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Min 23 Des 2020 09:15:05  WIB

from cf.baseCF import ExplainerBase
import torch
import numpy as np
import time
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DiCE(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        super().__init__(data_interface, model_interface) # initiating data related parameters

    def generate_counterfactuals(self, query_instance, total_CFs, desired_class="opposite", proximity_weight=0.5, diversity_weight=1.0, categorical_penalty=0.1, features_to_vary="all", yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist", optimizer="adam", learning_rate=0.05, min_iter=500, max_iter=5000, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_query_instance=True, tie_random=False, stopping_threshold=0.5): 

        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.

        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.

        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function. Takes "avg_dist" or "dpp_style:inverse_dist".
        :param optimizer: PyTorch optimization algorithm. Currently tested only with "pytorch:adam".

        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence. Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).

        """

        query_instance = self.data_interface.prepare_query(query_instance, normalized = True)
        query_instance = torch.FloatTensor(query_instance)
        
        mask = self.data_interface.get_mask_of_features_to_vary(features_to_vary)
        mask = torch.LongTensor(mask)
        
        # initialize the cf instances 
        if init_near_query_instance == False:
            if isinstance(self.data_interface.scaler, MinMaxScaler):
                cf_instances = torch.randn(total_CFs, query_instance.shape[1])
            elif isinstance(self.data_interface.scaler, StandardScaler):
                cf_instances = torch.rand(total_CFs, query_instance.shape[1])
            else:
                cf_instances = torch.randn(total_CFs, query_instance.shape[1])
        else: # initialize around the query instances
            cf_instances = query_instance.repeat(total_CFs, 1)
            for i in range(1, total_CFs):
                cf_instances[i] = cf_instances[i] + 0.01 * i
            #cf_instances += torch.randn(total_CFs, query_instance.shape[1]) * 0.1

        cf_instances = torch.FloatTensor(cf_instances)
        cf_instances = mask * cf_instances + (1 - mask) * query_instance
        
        self.total_CFs = total_CFs
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        self.feature_weights_list = np.ones((1, query_instance.shape[1]))
        mads = self.data_interface.get_mads(normalized = True)
        inverse_mads = 1 / mads
        inverse_mads = np.round(inverse_mads, 2)
        self.feature_weights_list = torch.from_numpy(inverse_mads)
        '''
        indices_features_to_vary = self.data_interface.get_indices_of_features_to_vary(features_to_vary)
        indices_features_to_vary = np.array([indices_features_to_vary])
        inverse_mads_selected = np.take_along_axis(inverse_mads[np.newaxis, :], indices_features_to_vary, 1)
        np.put_along_axis(self.feature_weights_list, indices_features_to_vary, inverse_mads_selected, 1)
        self.feature_weights_list = torch.from_numpy(self.feature_weights_list) # not equal
        '''

        # specify the optimizer 
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam([cf_instances], lr = learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop([cf_instances], lr = learning_rate)
        else:
            self.optimizer = torch.optim.Adam([cf_instances], lr = learning_rate)

        test_pred = self.model_interface.predict_tensor(query_instance)[0]

        if desired_class == "opposite":
            self.target_cf_class = 1.0 - torch.round(test_pred)

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        
        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random

        # running optimization steps
        start_time = time.time()

        # variables to backup best known CFs so far in the optimization process - if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = [0]*total_CFs
        self.best_backup_cfs_preds = [0]*total_CFs
        self.min_dist_from_threshold = 100

        # looping the find CFs depending on whether its random initialization or not
        iterations = 0
        loss_diff = 0
        prev_loss = 0
        while self.stop_loop(iterations, loss_diff, cf_instances) is False:
            
            cf_instances.requires_grad = True
            # zero all existing gradients
            self.optimizer.zero_grad()
            # get loss and backpropogate
            loss_value = self.compute_loss(query_instance, cf_instances, proximity_weight, diversity_weight, categorical_penalty)
            loss_value.backward(retain_graph = True)

            cf_instances.grad = cf_instances.grad * mask
            # update the variables
            self.optimizer.step()

            if verbose:
                if (iterations) % 50 == 0:
                    print('step %d,  loss=%g' % (iterations+1, loss_value))

            loss_diff = abs(loss_value-prev_loss)
            prev_loss = loss_value
            iterations += 1
    
        # clamp
        if isinstance(self.data_interface.scaler, MinMaxScaler):
            cf_instances = torch.where(cf_instances > 1, torch.ones_like(cf_instances), cf_instances)
            cf_instances = torch.where(cf_instances < 0, torch.zeros_like(cf_instances), cf_instances)

        return cf_instances

    def compute_yloss(self, cf_instances):
        yloss = 0
        if self.yloss_type == 'l2_loss':
            yloss = torch.pow((self.model_interface.predict_tensor(cf_instances)[1] - self.target_cf_class), 2)
        elif self.yloss_type == "log_loss":
            logits = torch.log(torch.abs(self.model_interface.predict_tensor(cf_instances)[1] - 1e-6) / torch.abs(1 - self.model_interface.predict_tensor(cf_instances)[1] - 1e-6))
            criterion = torch.nn.BCEWithLogitsLoss()
            yloss = criterion(logits, self.target_cf_class.repeat(self.total_CFs))

        elif self.yloss_type == 'hinge_loss':
            logits = torch.log(torch.abs(self.model_interface.predict_tensor(cf_instances)[1] - 1e-6) / torch.abs(1 - self.model_interface.predict_tensor(cf_instances)[1] - 1e-6))
            criterion = torch.nn.ReLU()
            all_ones = torch.ones_like(self.target_cf_class)
            labels = 2 * self.target_cf_class - all_ones
            temp_loss = all_ones - torch.mul(labels, logits)
            yloss = criterion(temp_loss)
            #y_loss = torch.norm(criterion(temp_loss), axis = 1)
    
        return yloss.mean()

    def compute_proximity_loss(self, query_instance, cf_instances):
        """compute weighted distance between query intances and counterfactual explanations"""
        return torch.mean(torch.abs(cf_instances - query_instance) * self.feature_weights_list)

    def compute_dist(self, x1, x2):
        return torch.sum(torch.mul(torch.abs(x1 - x2), self.feature_weights_list), dim = 0)

    def dpp_style(self, cf_instances, submethod):
        """Computes the DPP of a matrix."""

        det_entries = torch.ones(self.total_CFs, self.total_CFs)
        for i in range(self.total_CFs):
            for j in range(self.total_CFs):
                det_entries[i, j] = self.compute_dist(cf_instances[i], cf_instances[j])

        if submethod == "inverse_dist":
            det_entries = 1.0 / (1.0 + det_entries)
        if submethod == "exponential_dist":
            det_entries = 1.0 / (torch.exp(det_entries))

        det_entries += torch.eye(self.total_CFs) * 0.0001
        return torch.det(det_entries)

    def compute_diversity_loss(self, cf_instances):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(cf_instances, submethod)
        elif self.diversity_loss_type == "avg_dist":
            return 1 - 1 / (1.0 + self.mm(cf_instances, cf_instances.T))

    def compute_regularization_loss(self, cf_instances):
        """Adds a linear equality constraints to the loss functions - to ensure all levels of a categorical variable sums to one"""
        regularization_loss = 0
        for v in self.data_interface.encoded_categorical_feature_indices:
            regularization_loss += torch.sum(torch.pow((torch.sum(cf_instances[:, v[0]:v[-1]+1], axis = 1) - 1.0), 2))

        return regularization_loss

    def compute_loss(self, query_instance, cf_instances, proximity_weight, diversity_weight, categorical_penalty):
        """Computes the overall loss"""
        yloss = self.compute_yloss(cf_instances)
        proximity_loss = self.compute_proximity_loss(query_instance, cf_instances) if proximity_weight > 0 else 0.0
        diversity_loss = self.compute_diversity_loss(cf_instances) if diversity_weight > 0 else 0.0
        regularization_loss = self.compute_regularization_loss(cf_instances)

        loss = yloss + (proximity_weight * proximity_loss) - (diversity_weight * diversity_loss) + (categorical_penalty * regularization_loss)
        return loss

    def stop_loop(self, itr, loss_diff, cf_instances):
        """Determines the stopping condition for gradient descent."""

        if itr < self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= self.max_iter:
            return True

        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                test_preds = self.model_interface.predict_tensor(cf_instances)[1]
                if self.target_cf_class == 0 and (test_preds < self.stopping_threshold).all():
                    return True
                elif self.target_cf_class == 1 and (test_preds > self.stopping_threshold).all():
                    return True
                else:
                    return False
        else:
            self.loss_converge_iter = 0
            return False
