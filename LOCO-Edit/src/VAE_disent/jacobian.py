'''
1. how does channel A change when channel B is changed?
2. If i change the hidden state 'z', how do the channels A and B change?

there is no direct dA/dB: A and B are both decode(z) outputs, not functions of
each other. so we compute the two jacobians that DO exist,
    J_A = d(chA)/dz   and   J_B = d(chB)/dz,
and get everything else from them by chain rule.
'''
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import twoChannelVAE
from data_utils import twoChannelDataset

MODEL_PATH  = "vae_checkpoints/beta_vae_beta=1/vae_epoch99.pt"
LATENT_DIM  = 16
DEVICE      = torch.device("cuda:9")
CHA_DIR     = "data/singlecell_chA_split/train"
CHB_DIR     = "data/singlecell_chB_split/train"
OUT_DIR     = "vae_checkpoints/beta_vae_beta=1/jacobian_checks"
os.makedirs(OUT_DIR, exist_ok=True)

model = twoChannelVAE(latent_dim=LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

dataset = twoChannelDataset(CHA_DIR, CHB_DIR)


def decode_from_z(z):
    #z: [latent_dim] -> [2, H, W]. no no_grad here, we differentiate through it
    return model.decode(z.unsqueeze(0)).squeeze(0)


def full_jacobian(z):
    #[2, H, W, latent_dim] : per-pixel, per-channel derivative wrt each latent dim
    z = z.detach().clone().to(DEVICE).float().requires_grad_(True)
    J = torch.autograd.functional.jacobian(decode_from_z, z)
    return J.detach()   # [2, H, W, K]


def channel_dim_sensitivity(J, thresh=0.15):
    '''
    Q2: if i nudge z_k, how much does each channel move?
    reduce the pixel jacobian to a [2, K] magnitude matrix and classify each
    dim as A-specific / B-specific / shared(coupled).
    '''
    K = J.shape[-1]
    # L2 norm over pixels -> how strongly dim k drives each channel
    a = J[0].reshape(-1, K).norm(dim=0)   # [K]  ||dA/dz_k||
    b = J[1].reshape(-1, K).norm(dim=0)   # [K]  ||dB/dz_k||

    print("dim |  ||dA/dz|| | ||dB/dz|| |  A-share  | verdict")
    print("----+-----------+-----------+-----------+---------")
    for k in range(K):
        total = (a[k] + b[k]).item()
        share = a[k].item() / total if total > 1e-8 else 0.0   # ->1 A-only, ->0 B-only
        if total < 1e-6:
            verdict = "dead"
        elif share > 1 - thresh:
            verdict = "A-only"
        elif share < thresh:
            verdict = "B-only"
        else:
            verdict = "SHARED (A<->B coupled)"
        print(f" {k:2d} |  {a[k].item():8.4f} | {b[k].item():8.4f} |   {share:5.2f}   | {verdict}")
    return a, b


def save_jacobian_maps(J, out_dir=OUT_DIR):
    #spatial heatmap of sensitivity: for each dim, where in the image does chA / chB react
    K = J.shape[-1]
    for k in range(K):
        for ch, name in [(0, "chA"), (1, "chB")]:
            m = J[ch, :, :, k].abs()
            m = m / (m.max() + 1e-8)              # normalize per-map for viewing
            save_image(m.unsqueeze(0), f"{out_dir}/dim{k}_{name}_sensitivity.png")


def induced_change_in_A_from_B(J, target_B):
    '''
    Q1 (the "impossible" one, done through z):
    you want channel B to change by target_B. the least-norm latent step that
    achieves it is  dz = pinv(J_B) @ target_B.  that same dz drags A along by
    dA = J_A @ dz. the norm ratio ||dA|| / ||target_B|| is the A<-B coupling.
    '''
    P = target_B.numel()
    J_A = J[0].reshape(P, -1)          # [P, K]
    J_B = J[1].reshape(P, -1)          # [P, K]
    dB  = target_B.reshape(P, 1)       # [P, 1]

    dz = torch.linalg.pinv(J_B) @ dB   # [K, 1] : minimal z-move that yields dB
    dA = (J_A @ dz).reshape(target_B.shape)   # induced change in channel A

    coupling = dA.norm().item() / (dB.norm().item() + 1e-8)
    return dz.squeeze(1), dA, coupling


if __name__ == "__main__":
    #anchor on a real image, linearize the decoder at its latent code
    x_anchor = dataset[0].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mu_anchor, _ = model.encode(x_anchor)
    mu_anchor = mu_anchor.squeeze(0)   # [latent_dim]

    J = full_jacobian(mu_anchor)       # [2, H, W, K]
    print(f"jacobian shape: {tuple(J.shape)}  (channels, H, W, latent_dim)\n")

    print("== Q2: per-dim channel sensitivity ==")
    channel_dim_sensitivity(J)
    save_jacobian_maps(J)

    print("\n== Q1: A<->B coupling through z ==")
    #example target: brighten B exactly as its own current reconstruction
    with torch.no_grad():
        recon = decode_from_z(mu_anchor)   # [2, H, W]
    target_B = recon[1]                    # ask "make B look like this"
    dz, dA, coupling = induced_change_in_A_from_B(J, target_B)
    print(f"to drive B, latent step dz = {dz.cpu().numpy().round(3)}")
    print(f"induced |dA| / |dB| coupling = {coupling:.4f}")
    save_image((dA.abs() / (dA.abs().max() + 1e-8)).unsqueeze(0),
               f"{OUT_DIR}/induced_dA_from_B.png")