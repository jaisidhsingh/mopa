import math
import torch
from typing import *
from torch import Tensor

from src.utils import _assert_no_nan, _round_to_4_decimals
from src.moes.basic_moe import MoE


class ExpertChoiceMoE(MoE):
    def route_to_experts(self, x: Tensor, probs: Tensor) -> Tuple:
        B, T, D = x.shape
        N = self.n_experts
        device = x.device
        BT = B * T

        top1_val, top1_idx = probs.max(dim=-1)    # [B, T]
        x_flat = x.view(BT, D)
        eids = top1_idx.reshape(-1)               # [BT]
        scores = top1_val.reshape(-1)             # [BT]
        tok_ids = torch.arange(BT, device=device) # [BT]

        counts_req = torch.bincount(eids, minlength=N).float()
        expected = counts_req.new_full((N,), float(BT)/N)
        lb_loss = ((counts_req - expected)**2).mean() / (expected.clamp_min(1.0)**2).mean()

        C = self._capacity(BT, N, self.capacity_factor)

        y_flat = x_flat.clone()
        dropped = 0

        for e in range(N):
            mask = (eids == e)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]           # token positions targeting expert e
            sc  = scores.index_select(0, idx)
            
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
        _assert_no_nan(y, f"{self.name}.output")
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

