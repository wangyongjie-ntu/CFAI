#Filename:	baseCF.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 14 Des 2020 03:43:39  WIB

import numpy as np

class ExplainerBase(object):

    def __init__(self, data_interface, model_interface):

        self.data_interface = data_interface
        self.model_interface = model_interface

    def generate_counterfactuals(self):

        raise NotImplementedError

    def do_post_sparsity(self):

        raise NotImplementedError

    def do_post_filtering(self):

        raise NotImplementedError

  
