'''
DMVAE for 2 co-registered image channels (A = gene burst, B = condensate).
Each channel gets a PRIVATE latent; a SHARED latent is fused across both
channels via Product-of-Experts. Disentanglement is by construction:
  z_priv_A  -> factors unique to channel A
  z_priv_B  -> factors unique to channel B
  z_shared  -> factors common to both (the same physical cell)
Based on Lee & Pavlovic, DMVAE (CVPRW 2021), arXiv:2012.13024.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def kl_standard_normal(mu, logvar):
    #KL( N(mu, exp(logvar)) || N(0, I) ), summed over latent, mean over batch
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1).mean()


def product_of_experts(mus, logvars):
    '''
    Combine several Gaussian "experts" into one Gaussian.
    mus, logvars: [E, B, S]  (E experts, batch B, shared-dim S)
    The prior expert N(0, I) should already be stacked in as experts[0]
    for numerical stability.
    precision-weighted average:  var = 1/sum(1/var_i),  mu = var * sum(mu_i/var_i)
    '''
    var   = torch.exp(logvars)
    T     = 1.0 / var                     # precision of each expert
    T_sum = T.sum(dim=0)                  # [B, S]
    mu    = (mus * T).sum(dim=0) / T_sum  # [B, S]
    var_p = 1.0 / T_sum
    return mu, torch.log(var_p)


class Encoder(nn.Module):
    '''one channel (1, 128, 128) -> private (P) + shared (S) posterior params'''
    def __init__(self, priv_dim, shared_dim, ch=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch,  32, 4, 2, 1), nn.SiLU(),   # 128 -> 64
            nn.Conv2d(32,  64, 4, 2, 1), nn.SiLU(),   # 64  -> 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),   # 32  -> 16
            nn.Conv2d(128,256, 4, 2, 1), nn.SiLU(),   # 16  -> 8
            nn.Conv2d(256,256, 4, 2, 1), nn.SiLU(),   # 8   -> 4
        )
        self.flat = 256 * 4 * 4
        # outputs: [muP, logvarP, muS, logvarS]
        self.head = nn.Linear(self.flat, 2 * priv_dim + 2 * shared_dim)
        self.priv_dim, self.shared_dim = priv_dim, shared_dim

    def forward(self, x):
        h = self.body(x).flatten(1)
        h = self.head(h)
        P, S = self.priv_dim, self.shared_dim
        muP     = h[:, 0:P]
        logvarP = h[:, P:2*P]
        muS     = h[:, 2*P:2*P+S]
        logvarS = h[:, 2*P+S:2*P+2*S]
        return muP, logvarP, muS, logvarS


class Decoder(nn.Module):
    '''[private + shared] -> reconstruct one channel (1, 128, 128)'''
    def __init__(self, priv_dim, shared_dim, ch=1):
        super().__init__()
        self.fc = nn.Linear(priv_dim + shared_dim, 256 * 4 * 4)
        self.body = nn.Sequential(
            nn.ConvTranspose2d(256,256, 4, 2, 1), nn.SiLU(),  # 4  -> 8
            nn.ConvTranspose2d(256,128, 4, 2, 1), nn.SiLU(),  # 8  -> 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),  # 16 -> 32
            nn.ConvTranspose2d(64,  32, 4, 2, 1), nn.SiLU(),  # 32 -> 64
            nn.ConvTranspose2d(32,  ch, 4, 2, 1),             # 64 -> 128
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, 4, 4)
        return torch.tanh(self.body(h))   # data is normalized to [-1, 1]


class DMVAE(nn.Module):
    def __init__(self, priv_dim=8, shared_dim=8):
        super().__init__()
        self.priv_dim, self.shared_dim = priv_dim, shared_dim
        self.enc_A = Encoder(priv_dim, shared_dim)
        self.enc_B = Encoder(priv_dim, shared_dim)
        self.dec_A = Decoder(priv_dim, shared_dim)
        self.dec_B = Decoder(priv_dim, shared_dim)

    def fuse_shared(self, muS_A, logvarS_A, muS_B, logvarS_B):
        B = muS_A.shape[0]
        S = self.shared_dim
        device = muS_A.device
        #prior expert N(0, I): mu=0, logvar=0
        prior_mu     = torch.zeros(B, S, device=device)
        prior_logvar = torch.zeros(B, S, device=device)
        mus     = torch.stack([prior_mu,     muS_A,     muS_B],     dim=0)  # [3, B, S]
        logvars = torch.stack([prior_logvar, logvarS_A, logvarS_B], dim=0)
        return product_of_experts(mus, logvars)

    def forward(self, xA, xB):
        muP_A, lvP_A, muS_A, lvS_A = self.enc_A(xA)
        muP_B, lvP_B, muS_B, lvS_B = self.enc_B(xB)

        #shared posterior = PoE across both channels (+ prior)
        muS, lvS = self.fuse_shared(muS_A, lvS_A, muS_B, lvS_B)

        zP_A = reparameterize(muP_A, lvP_A)
        zP_B = reparameterize(muP_B, lvP_B)
        zS   = reparameterize(muS,   lvS)

        recon_A = self.dec_A(torch.cat([zP_A, zS], dim=1))
        recon_B = self.dec_B(torch.cat([zP_B, zS], dim=1))

        return {
            "recon_A": recon_A, "recon_B": recon_B,
            "priv_A": (muP_A, lvP_A), "priv_B": (muP_B, lvP_B),
            "shared": (muS, lvS),
        }

    #convenience for analysis later (Jacobian/SVD per block)
    @torch.no_grad()
    def encode(self, xA, xB):
        muP_A, _, muS_A, lvS_A = self.enc_A(xA)
        muP_B, _, muS_B, lvS_B = self.enc_B(xB)
        muS, _ = self.fuse_shared(muS_A, lvS_A, muS_B, lvS_B)
        return muP_A, muP_B, muS   # the three latent means