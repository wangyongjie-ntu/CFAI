#Filename:	VAE.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Jum 01 Jan 2021 10:31:50  WIB

import torch
import torch.nn as nn

class VAE(nn.Module):

    def __init__(self, data_size, 
            encoded_size,
            data_interface,
            hidden_dims = [20, 16, 12]
            ):

        super(VAE, self).__init__()

        self.data_size = data_size
        self.encoded_size = encoded_size
        self.data_interface = data_interface
        self.hidden_dims = hidden_dims

        modules = []
        
        in_channels = data_size
        #create encoder module
        for h_dim in self.hidden_dims:
            modules.append(
                    nn.Sequential(
                        nn.Linear(in_channels, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.Dropout(0.1),
                        nn.ReLU(),
                        )
                    )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(12, self.encoded_size)
        self.fc_var = nn.Linear(12, self.encoded_size)

        #create decoder module
        modules = []
        in_channels = encoded_size

        for h_dim in reversed(self.hidden_dims):
            modules.append(
                    nn.Sequential(
                        nn.Linear(in_channels, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.Dropout(0.1),
                        nn.ReLU(),
                        )
                    )
            in_channels = h_dim

        modules.append(nn.Linear(in_channels, self.data_size))
        self.sig = nn.Sigmoid()
        self.decoder = nn.Sequential(*modules)
    
    def encode(self, input_x):

        output = self.encoder(input_x)
        mu = self.fc_mu(output)
        log_var = self.fc_var(output)

        return [mu, log_var]

    def decode(self, z):

        x = self.decoder(z)
        for v in self.data_interface.encoded_categorical_feature_indices:    
            start_index = v[0]
            end_index = v[-1] + 1
            x[:,start_index:end_index] = self.sig(x[:,start_index:end_index])
        return x
    
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input_x):

        mu, log_var =  self.encode(input_x)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input_x, mu, log_var]

    def compute_loss(self, output, input_x, mu, log_var):

        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        con_criterion = nn.MSELoss()
        cat_criterion = nn.BCELoss()
        
        cat_loss = 0
        con_loss = 0
        
        for v in self.data_interface.encoded_categorical_feature_indices:
            start_index = v[0]
            end_index = v[-1]+1
            cat_loss += cat_criterion(output[:, start_index:end_index], input_x[:, start_index:end_index])
        
        categorial_indices = []
        for v in self.data_interface.encoded_categorical_feature_indices:
            categorial_indices.extend(v)

        continuous_indices = list(set(range(36)).difference(categorial_indices))
        con_loss = con_criterion(output[:, continuous_indices], input_x[:, continuous_indices])
        recon_loss = torch.mean(cat_loss + con_loss) 
        total_loss = kl_loss + recon_loss

        return total_loss, recon_loss, kl_loss
