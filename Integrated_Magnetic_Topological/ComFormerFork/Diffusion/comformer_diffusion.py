import torch
import torch_scatter
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from Diffusion.guassian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

from models.TI_conformer import eComformerConfig, eComformer

betas = torch.linspace(0.0001, 0.02, steps=1000).numpy()
lattice_diff = GaussianDiffusion(
    betas=betas,
    model_mean_type=ModelMeanType.EPSILON,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
)

betas = torch.linspace(0.0001, 0.02, steps=1000).numpy()
lattice_diff = GaussianDiffusion(
    betas=betas,
    model_mean_type=ModelMeanType.EPSILON,
    model_var_type=ModelVarType.FIXED_SMALL,
    loss_type=LossType.MSE,
)

def make_d3pm_transition_matrices(T, K, β_start=0.001, β_end=0.1):
    # Q_t = (1−β_t)·I + (β_t/K)·1·1^T
    betas = torch.linspace(β_start, β_end, T)
    I = torch.eye(K)
    ones = torch.ones(K, K)
    Q = [(1 - β)*I + (β/K)*ones for β in betas]
    return torch.stack(Q)  # shape (T, K, K)

K = 5 # num atom types in dataset
T = 1000 # num diffusion steps
species_Q = make_d3pm_transition_matrices(T, K)

def make_wrapped_normal_sigmas(T, sigma_min=1e-4, sigma_max=2.0, schedule="geometric"):
    """
    Build a schedule of T noise‐scales for wrapped‐normal diffusion.
    sigma_min = σ₁, sigma_max = σ_T.
    """
    if schedule == "linear":
        sigmas = np.linspace(sigma_min, sigma_max, T)
    elif schedule == "geometric":
        # geometric progression from sigma_min to sigma_max
        sigmas = np.exp(
            np.linspace(np.log(sigma_min), np.log(sigma_max), T)
        )
    else:
        raise ValueError(f"Unknown schedule {schedule}")
    return torch.from_numpy(sigmas).float()  # shape (T,)

coord_sigmas = make_wrapped_normal_sigmas(
    T=T,
    sigma_min=1e-3,
    sigma_max=2.0,
    schedule="geometric",
)


class JointDiffusion:
    def __init__(self, lattice_diff, coord_sigmas, species_Q):
        self.lattice = lattice_diff
        self.coord_sigmas = coord_sigmas
        self.species_Q = species_Q

    def q_sample_all(self, L0, F0, A0, t):
        """
        Diffuse each modality at time-step t:
          L0: (B,3,3) lattice
          F0: (B,N,3) fractional coords
          A0: (B,N) integer species labels
          t:  (B,) time indices
        Returns noisy (Lt, Ft, At) plus the coord noise for loss.
        """
        # 1) lattice
        Lt = self.lattice.q_sample(L0, t.cpu().numpy())

        # 2) coords
        σt = self.coord_sigmas[t]                 # (B,)
        noiseF = torch.randn_like(F0)
        Ft = F0 + σt.view(-1,1,1) * noiseF         # broadcast to (B,N,3)

        # 3) species
        Qt = self.species_Q[t]                    # (B,K,K)
        one_hot_A0 = F.one_hot(A0, K).float()      # (B,N,K)
        probs = one_hot_A0 @ Qt                    # (B,N,K)
        At = torch.multinomial(probs.view(-1, K), 1).view_as(A0)

        return Lt, Ft, At, noiseF

    def loss(self, model, L0, F0, A0, t):
        Lt, Ft, At, noiseF = self.q_sample_all(L0, F0, A0, t)
        epsL_hat, scoreF_hat, logitsA = model(Lt, Ft, At, t)

        # lattice ε‐prediction target
        α_bar = torch.from_numpy(self.lattice.alphas_cumprod).to(L0.device)
        σ_bar = torch.from_numpy(self.lattice.sqrt_one_minus_alphas_cumprod).to(L0.device)
        epsL_target = ((Lt - α_bar[t].view(-1,1,1)*L0) /
                       σ_bar[t].view(-1,1,1))

        # coord score‐matching target: ∇_Ft log 𝒩(Ft;F0,σ_t^2I) = −noiseF/σ_t
        σt = self.coord_sigmas[t].view(-1,1,1)
        scoreF_target = -noiseF / σt

        # species cross‐entropy
        loss_L = F.mse_loss(epsL_hat, epsL_target)
        loss_F = F.mse_loss(scoreF_hat, scoreF_target)
        loss_A = F.cross_entropy(logitsA.view(-1, K), A0.view(-1))

        return loss_L + loss_F + loss_A

joint = JointDiffusion(lattice_diff, coord_sigmas, species_Q)

class JointDiffusionTransformer(torch.nn.Module):
    def __init__(self, num_species: int, 
                 conv_config: eComformerConfig = eComformerConfig(name = "eComformer"),
    hidden_dim: int = 256):
        super().__init__()

        self.comformer = eComformer(conv_config)
        self.epsL_head = nn.Linear(conv_config.node_features, 9)
        self.scoreF_head = nn.Linear(conv_config.node_features, 3)
        self.species_head = nn.Linear(conv_config.node_features, num_species)

        self.time_embed = nn.Embedding(1000, conv_config.node_features)


    def forward(self, Lt, Ft, At, edge_index, edge_attr, batch, t):
        B = Lt.size(0)
        
        atom_embed = F.one_hot(At, num_classes=self.species_head.out_features).float()

        coord_embed = nn.Linear(3, atom_embed.size(1)).to(Ft.device)(Ft)

        # 3) time embedding (expand per node)
        time_vec = self.time_embed(t)                # (B, F)
        time_vec = time_vec[batch]                   # (B_total_nodes, F)

        # combine
        x = atom_embed + coord_embed + time_vec      # (B_total_nodes, F)

        # if you want, you can also inject a graph‐level "lattice embedding" into each node:
        L_flat = Lt.view(B, 9)
        L_embed = nn.Linear(9, x.size(1)).to(Lt.device)(L_flat)   # (B,F)
        x = x + L_embed[batch]

        # --- B) run ComformerConv layers ---
        # comformer expects a PyG‐style tuple (data, ldata, lattice)
        # Here `data` is (x, edge_index, edge_attr) and ignore lattice‐based equivariant update:
        node_feats = self.comformer((x, edge_index, edge_attr))

        # --- C) project into each head ---
        # 1) lattice ε (pool per‐graph then predict 3×3 noise)
        #    note: flatten back to (B,3,3)
        pooled = torch_scatter.scatter_mean(node_feats, batch, dim=0)  # (B, F)
        epsL_hat = self.epsL_head(pooled).view(B, 3, 3)

        # 2) coord score: (B_total_nodes,3)
        scoreF_hat = self.scoreF_head(node_feats)

        # 3) species logits: (B_total_nodes, K)
        logitsA = self.species_head(node_feats)

        return epsL_hat, scoreF_hat, logitsA

# Instantiate your model
model = JointDiffusionTransformer(num_species=K)