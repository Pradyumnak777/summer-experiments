'''
create a VAE (with high beta for diesntanglement) for 2 channel input
'''

import torch
import torch.nn as nn

class twoChannelVAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        #defining layers. input is a 2 channels of 128x128 pixels
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=4, stride=2, padding=1) #64 px
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1) #32 px
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1) #16 px
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) #8 px
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=256*8*8, out_features=256)
        # self.fc_hidden = nn.Linear(in_features=256, out_features=16)
        self.fc_mu = nn.Linear(in_features=256, out_features=self.latent_dim)
        self.fc_logvar = nn.Linear(in_features=256, out_features=self.latent_dim)
        self.leaky_relu = nn.LeakyReLU()
        
        '''
        decode layers below
        '''
        self.decoder_input = nn.Linear(self.latent_dim, 256 * 8 * 8)
        self.decode_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32 -> 64x64
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),    # 64x64 -> 128x128
            nn.Tanh()
        )
    def encode(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.leaky_relu(x)

        
        # latent = self.fc_hidden(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparametrize(self, mu, logvar):
        #latent is = mu + std * noise
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        

        return mu + eps * std
        
        
    def decode(self, z):
        #input is the 16 dim embedding
        x = self.decoder_input(z)
        x = torch.reshape(x,(-1, 256, 8, 8) )  #its [B, 256*8*8] before, now reshaped
        x = self.decode_layers(x)
        return x
    
    def forward(self, x):
        '''
        below for generral VAE
        # '''
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
        
        '''
        below for autoencoder test
        '''
        # mu, logvar = self.encode(x)
        # x_recon = self.decode(mu)
        # return x_recon, mu, logvar
    
    def loss(self, x, x_recon, mu, logvar, beta=4.0, mask=None, l1_weight=0.0):
        #shape is (Batch, 2, 128, 128)
        
        if mask is not None:
            se = (x_recon - x) ** 2 # [B, 2, H, W]
            ae = (x_recon - x).abs()
            mse_loss = (se * mask).sum()
            l1_loss  = (ae * mask).sum()
            # l1_loss  = ae.sum()
            # mse_loss = se.sum()

            
        else:
            # recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum') #ts is MSE
            mse_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
            l1_loss  = nn.functional.l1_loss(x_recon, x, reduction='sum')

        recon_loss = (1 - l1_weight) * mse_loss + l1_weight * l1_loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + (beta * kl_loss)
        
        batch_size = x.size(0)
        return total_loss / batch_size, recon_loss / batch_size, kl_loss / batch_size
        