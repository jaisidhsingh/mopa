import math
import torch
from abc import *
import torch.nn as nn
import torch.nn.functional as F

from models.utils import _assert_no_nan, _round_to_4_decimals, FeedForwardExpert


# ===========================
# Optimized Top-K Router
# ===========================
class TopKMoE_Fast(nn.Module):
    """
    Vectorized Top-K MoE with capacity.
    - No loops over tokens; only a single loop over experts to enforce capacity & run MLPs.
    Returns: y, aux_losses, diagnostics
    """
    def __init__(self, d_model:int, n_experts:int, d_hidden:int, k:int=2,
                 capacity_factor:float=1.25, normalize_gates:bool=True):
        super().__init__()
        assert 1 <= k <= n_experts and capacity_factor > 0
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.normalize_gates = normalize_gates

        self.experts = nn.ModuleList([FeedForwardExpert(d_model, d_hidden) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    @staticmethod
    def _capacity(tokens:int, n_experts:int, k:int, capacity_factor:float):
        return math.ceil(capacity_factor * (tokens * k) / n_experts)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        assert x.ndim == 3 and x.size(-1) == self.d_model
        B, T, D = x.shape
        N, K = self.n_experts, self.k
        device = x.device
        BT = B * T
        BTK = BT * K

        # ---- Gating (vectorized) ----
        logits = self.gate(x)                       # [B, T, N]
        probs  = F.softmax(logits, dim=-1)          # [B, T, N]
        _assert_no_nan(probs, "topk.probs")
        topk_vals, topk_idx = torch.topk(probs, k=K, dim=-1)   # [B, T, K]
        if self.normalize_gates:
            topk_vals = topk_vals / topk_vals.sum(-1, keepdim=True).clamp_min(1e-9)

        # Flatten token and replicate for K assignments
        x_flat = x.view(BT, D)                      # [BT, D]
        assign_e = topk_idx.reshape(-1)             # [BT*K] expert ids
        assign_w = topk_vals.reshape(-1)            # [BT*K] weights
        token_ids = torch.arange(BT, device=device).repeat_interleave(K)  # [BT*K]

        # ---- Load-balance loss (pre-capacity) ----
        total_assign = BT * K
        counts_pre = torch.bincount(assign_e, minlength=N).float()  # [N], sum up to BT*K
        expected = counts_pre.new_full((N,), float(total_assign)/N)
        lb_loss = ((counts_pre - expected)**2).mean() / (expected.clamp_min(1.0)**2).mean()

        # ---- Capacity ----
        C = self._capacity(BT, N, K, self.capacity_factor)

        # Prepare accumulators
        y_flat = torch.zeros(BT, D, device=device)
        contrib = torch.zeros(BT, 1, device=device)

        kept_total = 0
        # Single loop over experts (typically far smaller than BT)
        for e in range(N):
            mask = (assign_e == e)
            if not mask.any():
                continue

            idx_global = mask.nonzero(as_tuple=True)[0]          # positions in [BT*K]
            # rank assignments for this expert by gate weight (desc), keep top-C
            w_e = assign_w.index_select(0, idx_global)
            if idx_global.numel() > C:
                topw, topi = torch.topk(w_e, k=C, dim=0, largest=True, sorted=False)
                kept_idx_global = idx_global.index_select(0, topi)
                w_kept = topw
            else:
                kept_idx_global = idx_global
                w_kept = w_e

            # Gather inputs for this expert
            tok_kept = token_ids.index_select(0, kept_idx_global)     # [<=C]
            x_kept   = x_flat.index_select(0, tok_kept)               # [<=C, D]

            # Run expert and accumulate
            y_e = self.experts[e](x_kept)                             # [<=C, D]
            y_flat.index_add_(0, tok_kept, y_e * w_kept.unsqueeze(-1))
            contrib.index_add_(0, tok_kept, w_kept.unsqueeze(-1))
            kept_total += tok_kept.numel()

        # Fallback for tokens with no kept assignments (rare if C is sufficient)
        no_contrib = (contrib.squeeze(-1) <= 1e-12)
        if no_contrib.any():
            y_flat[no_contrib] = x_flat[no_contrib]

        y = y_flat.view(B, T, D)
        overflow = total_assign - kept_total
        expert_loads = (counts_pre / BTK).detach().cpu().tolist()

        aux = {
            "load_balance": lb_loss,
            "overflow": torch.tensor(float(overflow) / max(1, total_assign), device=device),
        }
        diag = {
            "capacity": C,
            "total_assignments": total_assign,
            "kept_assignments": int(kept_total),
            "overflow": int(overflow),
            "expert_loads": _round_to_4_decimals(expert_loads) 
        }
        _assert_no_nan(y, "topk.output")
        return y, aux, diag


# ===============================
# Optimized Expert-Choice Router
# ===============================
class ExpertChoiceMoE_Fast(nn.Module):
    """
    Vectorized Expert-Choice MoE with capacity.
    - Each token proposes scores; each expert chooses up to C top tokens.
    - Single loop over experts; everything else is vectorized.
    Returns: y, aux_losses, diagnostics
    """
    def __init__(self, d_model:int, n_experts:int, d_hidden:int, capacity_factor:float=1.25):
        super().__init__()
        assert d_model>0 and n_experts>0 and d_hidden>0 and capacity_factor>0
        self.d_model = d_model
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList([FeedForwardExpert(d_model, d_hidden) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    @staticmethod
    def _capacity(tokens:int, n_experts:int, capacity_factor:float):
        return math.ceil(capacity_factor * (tokens / n_experts))

    def forward(self, x):
        assert x.ndim == 3 and x.size(-1) == self.d_model
        B, T, D = x.shape
        N = self.n_experts
        device = x.device
        BT = B * T

        logits = self.gate(x)                     # [B, T, N]
        probs  = F.softmax(logits, dim=-1)        # [B, T, N]
        _assert_no_nan(probs, "ec.probs")

        top1_val, top1_idx = probs.max(dim=-1)    # [B, T]
        x_flat = x.view(BT, D)
        eids = top1_idx.reshape(-1)               # [BT]
        scores = top1_val.reshape(-1)             # [BT]
        tok_ids = torch.arange(BT, device=device) # [BT]

        # Load-balance (requests, pre-capacity)
        counts_req = torch.bincount(eids, minlength=N).float()
        expected = counts_req.new_full((N,), float(BT)/N)
        lb_loss = ((counts_req - expected)**2).mean() / (expected.clamp_min(1.0)**2).mean()

        C = self._capacity(BT, N, self.capacity_factor)

        # Default output = identity; overwrite the chosen tokens per expert
        y_flat = x_flat.clone()
        dropped = 0

        for e in range(N):
            mask = (eids == e)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]           # token positions targeting expert e
            sc  = scores.index_select(0, idx)
            # Keep top-C tokens for this expert
            if idx.numel() > C:
                topv, topi = torch.topk(sc, k=C, dim=0, largest=True, sorted=False)
                keep_idx = idx.index_select(0, topi)
                dropped += idx.numel() - C
            else:
                keep_idx = idx

            x_e = x_flat.index_select(0, keep_idx)         # [<=C, D]
            y_e = self.experts[e](x_e)                     # [<=C, D]
            y_flat.index_copy_(0, keep_idx, y_e)

        y = y_flat.view(B, T, D)
        _assert_no_nan(y, "ec.output")
        expert_loads = (counts_req / BT).detach().cpu().tolist()

        aux = {
            "load_balance": lb_loss,
            "overflow": torch.tensor(float(dropped) / max(1, BT), device=device),
        }
        diag = {
            "capacity": C,
            "requested_per_expert": counts_req.tolist(),
            "dropped_tokens": int(dropped),
            "expert_loads": _round_to_4_decimals(expert_loads)
        }
        return y, aux, diag
