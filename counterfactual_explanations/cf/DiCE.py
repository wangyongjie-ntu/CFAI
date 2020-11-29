"""
Module to generate diverse counterfactual explanations based on PyTorch framework
"""
from cf.explainer_base import ExplainerBase
import torch
import numpy as np
import random
import collections
import time
import copy

from cf import cf_display as cd

class DiCE(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """

        super().__init__(data_interface) # initiating data related parameters

        # initializing model variables
        self.model = model_interface

        # loading trained model
        self.model.load_model()

        # set the model in evaluation mode
        self.model.set_eval_mode()

        self.minx = torch.FloatTensor(self.data_interface.minx)
        self.maxx = torch.FloatTensor(self.data_interface.maxx)

    def generate_counterfactuals(self, query_instance, total_CFs, desired_target = 1.1, proximity_weight=0.5, diversity_weight=1.0, features_to_vary="all", yloss_type="l2_loss", diversity_loss_type="dpp_style:inverse_dist", feature_weights= None, optimizer="adam", learning_rate= 0.2, min_iter=500, max_iter=1000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False, init_near_query_instance=True, tie_random=False, posthoc_sparsity_param= None, posthoc_sparsity_algorithm="linear"):

        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.

        :param desired_target: Desired target compared with the current predicted target. Desired_target = test_pred * 1.1
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function. Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as values. Default option is "inverse_mad" where the weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set; the weight for a categorical feature is equal to 1 by default.
        :param optimizer: PyTorch optimization algorithm. Currently tested only with "pytorch:adam".

        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence. Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary". Prefer binary search when a feature range is large (for instance, income varying from 10k to 1000k) and only if the features share a monotonic relationship with predicted outcome in the model.

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations (see diverse_counterfactuals.py).

        """

        # check feature MAD validity and throw warnings
    
        start_time = time.time()
        self.total_CFs = total_CFs
        self.features_to_vary = features_to_vary
        self.features_to_vary_mask = torch.FloatTensor(self.data_interface.get_mask_of_features_to_vary(features_to_vary))
    
        query_instance = self.data_interface.normalize_data(query_instance)
        self.query_instance = torch.FloatTensor(query_instance)
        # initialize the N tota CFs
        self.do_cf_initializations(total_CFs, features_to_vary)

        if feature_weights == None:
            feature_weights = np.ones(len(self.data_interface.all_feature_names))
            self.feature_weights = torch.FloatTensor(feature_weights)
        else:
            feature_weights = self.data_interface.get_weights_of_features_to_vary(feature_weights)
            self.feature_weights = torch.FloatTensor(feature_weights)

        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight

        if optimizer == "adam":
            optim = torch.optim.Adam(self.cf_instances, learning_rate)
        else:
            optim = torch.optim.RMSprop(self.cf_instances, learning_rate)


        # find the predicted value of query_instance [1 * N]
        test_pred = self.predict_fn(query_instance).detach().numpy()
        raw_target = self.data_interface.de_normalize_target(test_pred)
        tmp = raw_target * desired_target
        target = self.data_interface.normalize_target(tmp)
        self.target = torch.FloatTensor(target) # tensor format of target

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False
        min_dist_from_threshold = 100

        # variables to backup best known CFs so far in the optimization process - if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
        self.best_backup_cfs = [0]*self.total_CFs
        self.best_backup_cfs_preds = [0]*self.total_CFs
        self.final_cfs = []

        # looping the find CFs depending on whether its random initialization or not

        iterations = 0
        loss_diff = 1.0
        prev_loss = 0.0

        while self.stop_loop(iterations, loss_diff) is False:


            # zero all existing gradients
            optim.zero_grad()

            # get loss and backpropogate
            loss_value = self.compute_loss()
            loss_value.backward()

            # freeze features other than feat_to_vary_idxs
            for idx in range(self.total_CFs):
                self.cf_instances[idx].grad = self.cf_instances[idx].grad * self.features_to_vary_mask

            # update the variables
            optim.step()

            # element-wise clamp for each cf
            for idx in range(self.total_CFs):
                self.cf_instances[idx] = torch.where(self.cf_instances[idx] > self.maxx, self.maxx, self.cf_instances[idx])
                self.cf_instances[idx] = torch.where(self.cf_instances[idx] < self.minx, self.minx, self.cf_instances[idx])
                self.cf_instances[idx].detach_()
                self.cf_instances[idx].requires_grad_()

            if verbose:
                if (iterations) % 50 == 0:
                    print('step %d,  loss=%g' % (iterations+1, loss_value))

            loss_diff = abs(loss_value-prev_loss)
            prev_loss = loss_value
            iterations += 1

            # backing up CFs if they are valid
            temp_cfs_stored = self.round_off_cfs(assigned = False)
            temp_test_preds_stored = [self.predict_fn(cf) for cf in temp_cfs_stored]

            if(all(i >= self.target for i in temp_test_preds_stored)): 
                avg_preds_dist = np.mean([(self.target - pred).item() for pred in temp_test_preds_stored])
                if avg_preds_dist < min_dist_from_threshold:
                    min_dist_from_threshold = avg_preds_dist
                    for idx in range(self.total_CFs):
                        self.best_backup_cfs[idx] = temp_cfs_stored[idx].clone()
                        self.best_backup_cfs_preds[idx] = temp_test_preds_stored[idx].clone()

        # rounding off final cfs - not necessary when inter_project=True
        self.round_off_cfs(assigned = True)

        # storing final CFs
        for j in range(0, self.total_CFs):
            temp = self.cf_instances[j]
            self.final_cfs.append(temp)

        # max iterations at which GD stopped
        self.max_iterations_run = iterations

        self.elapsed = time.time() - start_time
        self.final_cf_instances_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # Convert the final cfs to numpy 
        self.final_cfs = [i.detach().numpy() for i in self.final_cfs]
        self.final_cf_instances_preds = [i.detach().numpy() for i in self.final_cf_instances_preds]

        # update final_cfs from backed up CFs if valid CFs are not found - currently works for DiverseCF only
        self.valid_cfs_found = False
        if (any(i <= target for i in self.final_cf_instances_preds)):
            if min_dist_from_threshold < 100:
                for ix in range(self.total_CFs):
                    self.final_cfs[ix] = self.best_backup_cfs[ix].detach().numpy()
                    self.final_cf_instances_preds[ix] = self.best_backup_cfs_preds[ix].detach().numpy()

                self.valid_cfs_found = True # final_cfs have valid CFs through backup CFs
            else:
                self.valid_cfs_found = False # neither final_cfs nor backup cfs are valid
        else:
            self.valid_cfs_found = True # final_cfs have valid CFs

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_sparse = copy.deepcopy(self.final_cfs)
            cfs_preds_sparse = copy.deepcopy(self.final_cf_instances_preds)
            self.final_cfs_sparse, self.cf_instances_preds_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_sparse, cfs_preds_sparse,  query_instance, posthoc_sparsity_param, posthoc_sparsity_algorithm)
        else:
            self.final_cfs_sparse = None
            self.cf_instances_preds_sparse = None

        m, s = divmod(self.elapsed, 60)
        if self.valid_cfs_found:
            self.total_CFs_found = self.total_CFs
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            '''
            self.total_CFs_found = 0
            for pred in self.cf_instances_preds:
                if((self.target_cf_class == 0 and pred[0][0] < self.stopping_threshold) or (self.target_cf_class == 1 and pred[0][0] > self.stopping_threshold)):
                    self.total_CFs_found += 1

            print('Only %d (required %d) Diverse Counterfactuals found for the given configuation, perhaps try with different values of proximity (or diversity) weights or learning rate...' % (self.total_CFs_found, self.total_CFs), '; total time taken: %02d' % m, 'min %02d' % s, 'sec')  

            '''
        
        self.returns = []
        for i in range(4):
            tmp = self.final_cfs[i]
            result = self.data_interface.de_normalize_data(tmp)
            result = np.round(result, 2)
            self.returns[i] = result
        
        return results


        return self.final_cfs

    def round_off_cfs(self, assigned = False, precision = 2):
        """ function for intermediate projection of CFs"""

        temp_cfs = []
        for idx in range(len(self.cf_instances)):
            cf = self.cf_instances[idx].detach().numpy()
            org_content = self.data_interface.de_normalize_data(cf[np.newaxis, :])
            org_content = np.round(org_content, precision)
            norm_cf = self.data_interface.normalize_data(org_content).squeeze(0)
            temp_cfs.append(torch.FloatTensor(norm_cf))
            if assigned:
                tmp = torch.FloatTensor(norm_cf)
                tmp.requires_grad_()
                self.cf_instances[idx] = tmp

        if not assigned:
            return temp_cfs
        else:
            return None

    def predict_fn(self, input_instance):
        """prediction function"""
        if not torch.is_tensor(input_instance):
            input_instance = torch.tensor(input_instance).float()

        return self.model.get_output(input_instance)

    def do_cf_initializations(self, total_CFs, features_to_vary):
        """Intializes CFs and other related variables."""

        # CF initialization. Initialize around current instance
        self.cf_instances = []
        for ix in range(self.total_CFs):
            one_init = np.random.uniform(self.data_interface.minx, self.data_interface.maxx)
            one_init = torch.FloatTensor(one_init)
            one_init = one_init * self.features_to_vary_mask + (1 - self.features_to_vary_mask) * self.query_instance.squeeze(0)
            one_init.requires_grad_()
            self.cf_instances.append(one_init)

    def compute_yloss(self):
        yloss = 0
        for i in range(self.total_CFs):
            if self.yloss_type == "l2_loss":
                temp_loss = torch.pow((self.predict_fn(self.cf_instances[i]) - self.target), 2)
            elif self.yloss_type == "log_loss":
                temp_logits = torch.log((abs(self.predict_fn(self.cf_instances[i]) - 0.000001))/(1 - abs(self.predict_fn(self.cf_instances[i]) - 0.000001)))
                criterion = torch.nn.BCEWithLogitsLoss()
                temp_loss = criterion(temp_logits, torch.tensor([self.target_cf_class]))
            elif self.yloss_type == "hinge_loss":
                temp_logits = torch.log((abs(self.predict_fn(self.cf_instances[i]) - 0.000001))/(1 - abs(self.predict_fn(self.cf_instances[i]) - 0.000001)))
                criterion = torch.nn.ReLU()
                all_ones = torch.ones_like(self.target_cf_class)
                labels = 2 * self.target_cf_class - all_ones
                temp_loss = all_ones - torch.mul(labels, temp_logits)
                temp_loss = torch.norm(criterion(temp_loss))

            yloss += temp_loss

        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights), dim=0)

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(self.cf_instances[i], self.query_instance.squeeze(0))
        return proximity_loss/(torch.mul(len(self.minx), self.total_CFs))

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = torch.ones((self.total_CFs, self.total_CFs))
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i,j)] = 1.0/(1.0 + self.compute_dist(self.cf_instances[i], self.cf_instances[j]))
                    if i == j:
                        det_entries[(i,j)] += 0.0001

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i,j)] = 1.0/(torch.exp(self.compute_dist(self.cf_instances[i], self.cf_instances[j])))
                    if i == j:
                        det_entries[(i,j)] += 0.0001

        diversity_loss = torch.det(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(submethod)
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cf_instances[i], self.cf_instances[j]))

            return 1.0 - (diversity_loss/count)

    def compute_loss(self):
        """Computes the overall loss"""
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0

        return self.yloss + (self.proximity_weight * self.proximity_loss) - (self.diversity_weight * self.diversity_loss)

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            for i in range(len(self.minx)):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        self.cf_instances[n].data[i] = query_instance[i]+(n*0.01)
                    else:
                        self.cf_instances[n].data[i] = np.random.uniform(self.minx[i], self.maxx[i])
                else:
                    self.cf_instances[n].data[i] = query_instance[i]

    def stop_loop(self, itr, loss_diff):
        """Determines the stopping condition for gradient descent."""

        # intermediate projections
        if((self.project_iter > 0) & (itr > 0)):
            if((itr % self.project_iter) == 0):
                self.round_off_cfs(assigned=True)

        # do GD for min iterations
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
                temp_cfs = self.round_off_cfs(assigned=False)
                test_preds = [self.predict_fn(cf)[0] for cf in temp_cfs]

                if all(i >= self.target for i in test_preds):
                    self.converged = True
                    return True
                else:
                    return False
        else:
            self.loss_converge_iter = 0
            return False


