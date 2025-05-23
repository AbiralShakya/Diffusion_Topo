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

    # def loss(self, model, L0, F0, A0, t):
    #     Lt, Ft, At, noiseF = self.q_sample_all(L0, F0, A0, t)
        
    #     epsL_hat, scoreF_hat, logitsA = model(Lt, Ft, At, t)

    #     # lattice ε‐prediction target
    #     α_bar = torch.from_numpy(self.lattice.alphas_cumprod).to(L0.device)
    #     σ_bar = torch.from_numpy(self.lattice.sqrt_one_minus_alphas_cumprod).to(L0.device)
    #     epsL_target = ((Lt - α_bar[t].view(-1,1,1)*L0) /
    #                    σ_bar[t].view(-1,1,1))

    #     # coord score‐matching target: ∇_Ft log 𝒩(Ft;F0,σ_t^2I) = −noiseF/σ_t
    #     σt = self.coord_sigmas[t].view(-1,1,1)
    #     scoreF_target = -noiseF / σt

    #     # species cross‐entropy
    #     loss_L = F.mse_loss(epsL_hat, epsL_target)
    #     loss_F = F.mse_loss(scoreF_hat, scoreF_target)
    #     loss_A = F.cross_entropy(logitsA.view(-1, K), A0.view(-1))

    #     return loss_L + loss_F + loss_A

    # In JointDiffusion class:
# def loss(self, model, L0, F0, A0, t): # Original
# Needs to become something like:
    def loss(self, model, L0, F0, A0, edge_index_0, edge_attr_0, batch_idx_0, t):
        Lt_batch, Ft_batch, At_batch, noiseF_batch = self.q_sample_all(L0, F0, A0, t)
       
        epsL_hat, scoreF_hat, logitsA = model(Lt_batch, Ft_batch, At_batch,
                                            edge_index_0, edge_attr_0, batch_idx_0, t)

        # lattice ε‐prediction target
        # Ensure L0 and Lt_batch are properly aligned for batch ops
        α_bar = torch.from_numpy(self.lattice.alphas_cumprod).to(L0.device)[t] # (B,)
        σ_bar = torch.from_numpy(self.lattice.sqrt_one_minus_alphas_cumprod).to(L0.device)[t] # (B,)
        # Expand for (B,3,3) operations
        epsL_target = ((Lt_batch - α_bar.view(-1,1,1)*L0) /
                    σ_bar.view(-1,1,1))

        # coord score‐matching target: ∇_Ft log 𝒩(Ft;F0,σ_t^2I) = −noiseF/σ_t
        # σt comes from self.coord_sigmas[t] which should be (B,)
        # noiseF_batch is (total_nodes,3)
        # σt needs to be expanded for nodes
        
        σt_expanded = self.coord_sigmas.to(F0.device)[t][batch_idx_0].view(-1,1) # (total_nodes, 1)
        # if Ft is (B,N,3), then σt.view(-1,1,1) is correct. If Ft is (total_nodes,3), adapt σt indexing.
        # Original code for Ft used view(-1,1,1) which implies Ft was (B,N,3) in q_sample_all.
        # If Ft_batch is (total_nodes, 3), then σt needs to be indexed by t and then by batch_idx_0
        # to align with each node.
        # Assuming Ft_batch from q_sample_all is (B, N_max, 3) if padded, or you handle variable N carefully.
        # Let's assume Ft and noiseF are (B,N,3) as in the original snippet's `q_sample_all` for Ft.
        # Then σt.view(-1,1,1) is correct for element-wise ops if σt is (B,).

        # Corrected σt application for scoreF_target, assuming F0 and noiseF are (B,N,3)
        # and t is (B,)
        σt_per_sample = self.coord_sigmas.to(F0.device)[t] # Shape (B,)
        scoreF_target = -noiseF_batch / σt_per_sample.view(-1,1,1) # Broadcasts (B,) to (B,N,3)

        # species cross‐entropy
        # logitsA: (total_nodes, K), A0_batch: (total_nodes,)
        loss_L = F.mse_loss(epsL_hat, epsL_target)
        loss_F = F.mse_loss(scoreF_hat, scoreF_target) # scoreF_hat should be (total_nodes, 3) or (B,N,3)
        loss_A = F.cross_entropy(logitsA.view(-1, K), A0.view(-1)) # A0 here should be the batched, flattened A0 corresponding to logitsA

        return loss_L + loss_F + loss_A

#joint = JointDiffusion(lattice_diff, coord_sigmas, species_Q)

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
    


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = JointDiffusionTransformer(num_species=K).to(device)
# joint_diffusion_obj = JointDiffusion(lattice_diff, coord_sigmas, species_Q)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Example optimizer

# num_epochs = ...
# train_loader = ... (your DataLoader)

# model.train()
# for epoch in range(num_epochs):
#     for batch_data in train_loader:
#         # Assuming batch_data is a dict or tuple from your DataLoader
#         # L0, F0, A0, edge_index, edge_attr, batch_map_nodes = ... unpack and move to device
#         L0 = batch_data['L0'].to(device)
#         F0 = batch_data['F0'].to(device) # e.g., (B, N_max, 3) padded or (total_nodes, 3)
#         A0 = batch_data['A0'].to(device) # e.g., (B, N_max) or (total_nodes,)
#         edge_index = batch_data['edge_index'].to(device)
#         edge_attr = batch_data['edge_attr'].to(device)
#         batch_idx = batch_data['batch_idx'].to(device) # Maps nodes to graph index in batch

#         optimizer.zero_grad()

#         # Sample timesteps for each item in the batch
#         # t should have shape (B,) where B is the batch size.
#         t = torch.randint(0, T, (L0.size(0),), device=device).long()

#         # Calculate loss using the modified loss function
#         # The shapes of F0, A0 passed to loss must match what q_sample_all expects.
#         # If using total_nodes representation, you'll need to adjust how L0/Lt and t are handled for batching.
#         # The original code assumes L0 is (B,3,3) and t is (B,).
#         # For F0 (B,N,3), A0 (B,N), this is consistent.
#         total_loss = joint_diffusion_obj.loss(model, L0, F0, A0, edge_index, edge_attr, batch_idx, t)

#         total_loss.backward()
#         optimizer.step()

#         # Log loss, etc.
#     # print(f"Epoch {epoch}, Loss: {total_loss.item()}")
    # Save checkpoints


    # model.eval()
# Lt_curr, Ft_curr, At_curr = Lt_T, Ft_T, At_T
# # Precompute \bar{Q}_t for D3PM species sampling if needed
# # Q_bar_matrices = ... # Compute cumulative products of species_Q

# for time_step in range(T - 1, -1, -1):
#     t = torch.full((B_gen,), time_step, device=device, dtype=torch.long)

#     # Recompute graph structure if dynamic, or use a fixed one
#     # edge_index_curr, edge_attr_curr = compute_graph(Ft_curr, batch_idx_gen)

#     with torch.no_grad():
#         epsL_hat, scoreF_hat, logitsA = model(Lt_curr, Ft_curr, At_curr,
#                                               edge_index_curr, edge_attr_curr, batch_idx_gen, t)

#     # 1. Denoise Lattice (using p_sample from GaussianDiffusion)
#     # This typically involves: L_{t-1} = (1/sqrt(alpha_t)) * (Lt - (beta_t / sqrt(1-alpha_bar_t)) * epsL_hat) + sigma_t * z_L
#     # Refer to your GaussianDiffusion implementation for the exact p_sample step.
#     Lt_curr = lattice_diff.p_sample(model_output=epsL_hat, x=Lt_curr, t=t.cpu()).to(device) # Adapt based on your p_sample signature

#     # 2. Denoise Coordinates (Score-based sampling)
#     # noiseF_pred = -scoreF_hat * joint_diffusion_obj.coord_sigmas[t].view(-1,1,1).to(device)
#     # F0_pred = Ft_curr - joint_diffusion_obj.coord_sigmas[t].view(-1,1,1).to(device) * noiseF_pred
#     # Or, more generally, use a score-based SDE/ODE solver step.
#     # E.g., Euler-Maruyama for reverse SDE:
#     # sigma_t = joint_diffusion_obj.coord_sigmas[t].view(-1,1,1).to(device)
#     # sigma_t_minus_1 = joint_diffusion_obj.coord_sigmas[t-1].view(-1,1,1).to(device) if time_step > 0 else torch.zeros_like(sigma_t)
#     # F_drift = -sigma_t**2 * scoreF_hat
#     # F_diffusion = sigma_t * sqrt_delta_t * torch.randn_like(Ft_curr) (delta_t needs careful definition)
#     # A common approach from score SDE papers (approximate, assumes t is continuous 0..1, sigmas defined on that):
#     # dt = -1.0 / T
#     # Ft_curr = Ft_curr + (sigma_t**2) * scoreF_hat * (-dt) + sigma_t * np.sqrt(-dt) * torch.randn_like(Ft_curr)
#     # This part requires careful implementation based on the precise nature of coord_sigmas and chosen sampler.
#     # For simplicity, let's denote a conceptual step:
#     Ft_curr = sample_coordinates_step(Ft_curr, scoreF_hat, joint_diffusion_obj.coord_sigmas, t, device) # You'll need to implement this

#     # 3. Denoise Species (D3PM reverse step)
#     # This involves:
#     #   a. Get p(A0_hat | Lt, Ft, At, t) = softmax(logitsA)
#     #   b. Sample A0_hat from this distribution.
#     #   c. Sample A_{t-1} from q(A_{t-1} | At_curr, A0_hat) using species_Q and precomputed Q_bar.
#     # This is complex. For a simpler start (but less accurate), you might sample A0 directly from logitsA at each step.
#     At_curr = sample_species_step(At_curr, logitsA, joint_diffusion_obj.species_Q, Q_bar_matrices, t, device) # Implement this

# # Lt_gen, Ft_gen, At_gen = Lt_curr, Ft_curr, At_curr are the generated structures