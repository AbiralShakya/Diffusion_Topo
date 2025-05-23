import torch
import torch.nn.functional as F
from Diffusion.guassian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

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

K = 5
species_Q = make_d3pm_transition_matrices(T, K)


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
    def __init__(self, num_species, d_model=128, n_layers=6, heads=4):
        super().__init__()
        # -- your point‐cloud / graph Transformer goes here.
        # It should ingest (Lt, Ft, At, t) and output:
        #   * epsL_hat:   (B,3,3)
        #   * scoreF_hat: (B,N,3)
        #   * logitsA:    (B,N,K)
        
    def forward(self, Lt, Ft, At, t):
        # embed time‐step t, run self‐attention over the N points, 
        # incorporate lattice via relative‐pos / spherical embeddings, etc.
        return epsL_hat, scoreF_hat, logitsA

# Instantiate your model
model = JointDiffusionTransformer(num_species=K)