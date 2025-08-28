import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import _assert_no_nan, _round_to_4_decimals, FeedForwardExpert


class TopKMoE(nn.Module):
    """
    Top-K Mixture-of-Experts with capacity limits and aux losses.
    Returns: y, aux_losses(dict), diagnostics(dict)
    """
    def __init__(self, d_model:int, n_experts:int, d_hidden:int, k:int=2,
                 capacity_factor:float=1.25, normalize_gates:bool=True):
        super().__init__()
        assert d_model>0 and n_experts>0 and d_hidden>0
        assert 1 <= k <= n_experts
        assert capacity_factor>0
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.normalize_gates = normalize_gates

        self.experts = nn.ModuleList([FeedForwardExpert(d_model, d_hidden) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    @staticmethod
    def _capacity(tokens:int, n_experts:int, k:int, capacity_factor:float):
        # Common heuristic: proportional to total assignments tokens*k divided by experts
        return math.ceil(capacity_factor * (tokens * k) / n_experts)

    def forward(self, x):
        # x: [B, T, D]
        assert x.ndim == 3, "x must be [batch, seq, d_model]"
        B, T, D = x.shape
        assert D == self.d_model
        N = self.n_experts
        device = x.device
        BT = B * T
        BTK = BT * self.k

        # Gating
        logits = self.gate(x)                       # [B, T, N]
        _assert_no_nan(logits, "gating.logits")
        probs = F.softmax(logits, dim=-1)           # [B, T, N]
        _assert_no_nan(probs, "gating.probs")

        # top-k selection per token
        topk_vals, topk_idx = torch.topk(probs, k=self.k, dim=-1)  # [B, T, k]
        if self.normalize_gates:
            denom = topk_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            topk_vals = topk_vals / denom

        total_assignments = B*T*self.k
        C = self._capacity(B*T, N, self.k, self.capacity_factor)

        x_flat = x.reshape(B*T, D)                  # [BT, D]
        _assert_no_nan(x_flat, "x_flat")
        topk_vals_f = topk_vals.reshape(B*T, self.k)
        topk_idx_f = topk_idx.reshape(B*T, self.k)

        # Load-balance loss (pre-capacity)
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx_f, num_classes=N).float()   # [BT, k, N]
            pre_counts = one_hot.sum(dim=1).sum(dim=0)               # [N]
            expected = torch.full_like(pre_counts, float(total_assignments)/N)
            lb_loss = ((pre_counts - expected)**2).mean() / (expected.clamp_min(1.0)**2).mean()

        # Build capacity-bounded queues per expert (clear and explicit for verification)
        expert_inputs = [torch.empty((0, D), device=device) for _ in range(N)]
        expert_gates  = [torch.empty((0,), device=device) for _ in range(N)]
        expert_orig_indices = [torch.empty((0,), dtype=torch.long, device=device) for _ in range(N)]

        for pos in range(B*T):
            for j in range(self.k):
                e = int(topk_idx_f[pos, j].item())
                w = float(topk_vals_f[pos, j].item())
                if expert_inputs[e].shape[0] < C:
                    expert_inputs[e] = torch.cat([expert_inputs[e], x_flat[pos:pos+1, :]], dim=0)
                    expert_gates[e]  = torch.cat([expert_gates[e], torch.tensor([w], device=device)], dim=0)
                    expert_orig_indices[e] = torch.cat([expert_orig_indices[e], torch.tensor([pos], device=device)])

        kept = sum(inp.shape[0] for inp in expert_inputs)
        overflow = total_assignments - kept
        overflow_frac = overflow / max(1, total_assignments)

        # Run experts
        expert_outputs = []
        for e in range(N):
            y_e = self.experts[e](expert_inputs[e]) if expert_inputs[e].shape[0] > 0 else torch.empty((0, D), device=device)
            expert_outputs.append(y_e)

        # Combine back to [BT, D]
        y_flat = torch.zeros(B*T, D, device=device)
        contrib_sum = torch.zeros(B*T, 1, device=device)
        for e in range(N):
            if expert_outputs[e].shape[0] == 0: continue
            idxs = expert_orig_indices[e].long()
            w = expert_gates[e].unsqueeze(-1)
            _assert_no_nan(w, f"expert{e}.gates")
            y_flat.index_add_(0, idxs, expert_outputs[e] * w)
            contrib_sum.index_add_(0, idxs, w)

        # fallback for any token with zero kept assignments
        no_contrib_mask = (contrib_sum.squeeze(-1) <= 1e-12)
        if no_contrib_mask.any():
            y_flat[no_contrib_mask] = x_flat[no_contrib_mask]

        y = y_flat.reshape(B, T, D)
        _assert_no_nan(y, "moe.output")
        expert_loads = (pre_counts / BTK).detach().cpu().tolist()

        aux_losses = {
            "load_balance": lb_loss, 
            "overflow": torch.tensor(float(overflow_frac), device=device),
            "expert_loads": _round_to_4_decimals(expert_loads)
        }
        diagnostics = {
            "capacity": C, 
            "total_assignments": total_assignments, 
            "kept_assignments": kept, 
            "overflow": overflow
        }
        return y, aux_losses, diagnostics

class ExpertChoiceMoE(nn.Module):
    """
    Expert-Choice MoE (EC): tokens propose scores; each expert chooses up to C top tokens.
    Dropped tokens pass through identity. Returns y, aux_losses, diagnostics.
    """
    def __init__(self, d_model:int, n_experts:int, d_hidden:int, capacity_factor:float=1.25):
        super().__init__()
        assert d_model>0 and n_experts>0 and d_hidden>0
        self.d_model = d_model
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        self.experts = nn.ModuleList([FeedForwardExpert(d_model, d_hidden) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    @staticmethod
    def _capacity(tokens:int, n_experts:int, capacity_factor:float):
        return math.ceil(capacity_factor * (tokens / n_experts))

    def forward(self, x):
        assert x.ndim == 3
        B, T, D = x.shape
        assert D == self.d_model
        N = self.n_experts
        device = x.device

        logits = self.gate(x)                      # [B, T, N]
        probs = F.softmax(logits, dim=-1)          # [B, T, N]
        _assert_no_nan(probs, "ec.probs")

        # token's preferred expert (top-1)
        top1_val, top1_idx = probs.max(dim=-1)     # [B, T]
        BT = B*T
        C = self._capacity(BT, N, self.capacity_factor)

        # list tokens per expert
        x_flat = x.reshape(BT, D)
        top1_idx_f = top1_idx.reshape(BT)
        top1_val_f = top1_val.reshape(BT)

        per_expert_tokens = [[] for _ in range(N)]
        for pos in range(BT):
            e = int(top1_idx_f[pos].item())
            score = float(top1_val_f[pos].item())
            per_expert_tokens[e].append((pos, score))

        # each expert keeps up to C highest-score tokens
        expert_inputs, expert_orig_indices = [], []
        drops = 0
        for e in range(N):
            tokens = per_expert_tokens[e]
            if tokens:
                tokens.sort(key=lambda t: t[1], reverse=True)
            kept = tokens[:C]
            dropped = len(tokens) - len(kept)
            drops += max(0, dropped)
            idxs = torch.tensor([t[0] for t in kept], dtype=torch.long, device=device) if kept else torch.empty((0,), dtype=torch.long, device=device)
            expert_orig_indices.append(idxs)
            expert_inputs.append(x_flat.index_select(0, idxs) if kept else torch.empty((0, D), device=device))

        # run experts
        expert_outputs = []
        for e in range(N):
            y_e = self.experts[e](expert_inputs[e]) if expert_inputs[e].shape[0] > 0 else torch.empty((0, D), device=device)
            expert_outputs.append(y_e)

        # scatter back, identity for drops
        y_flat = x_flat.clone()
        for e in range(N):
            if expert_outputs[e].shape[0] == 0: continue
            idxs = expert_orig_indices[e]
            y_flat.index_copy_(0, idxs, expert_outputs[e])

        y = y_flat.reshape(B, T, D)
        _assert_no_nan(y, "ec.output")

        with torch.no_grad():
            counts = torch.tensor([len(per_expert_tokens[e]) for e in range(N)],
                                  dtype=torch.float32, device=device)
            expected = torch.full_like(counts, float(BT)/N)
            lb_loss = ((counts - expected)**2).mean() / (expected.clamp_min(1.0)**2).mean()
            overflow_frac = float(drops) / max(1, BT)
            expert_loads = (counts / BT).detach().cpu().tolist()

        aux_losses = {
            "load_balance": lb_loss, 
            "overflow": torch.tensor(overflow_frac, device=device),
            "expert_loads": _round_to_4_decimals(expert_loads) 
        }
        diagnostics = {
            "capacity": C, 
            "requested_per_expert": counts.tolist(), 
            "dropped_tokens": int(drops)
        }
        return y, aux_losses, diagnostics

