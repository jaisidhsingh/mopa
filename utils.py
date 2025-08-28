import torch
import torch.nn as nn
import torch.nn.functional as F


def _round_to_4_decimals(list_of_floats):
    return [round(item, 4) for item in list_of_floats]

def _assert_no_nan(t, name): 
    assert torch.isfinite(t).all(), f"{name} contains NaN/Inf"

class FeedForwardExpert(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden, bias=True)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=True)

    def forward(self, x):
        _assert_no_nan(x, "expert.input")
        return self.fc2(F.gelu(self.fc1(x)))

