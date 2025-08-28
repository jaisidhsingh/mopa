import tyro
import torch
from copy import deepcopy
from time import perf_counter
from dataclasses import dataclass

from moes import *
from fast_moes import *


@dataclass
class TestingConfig:
    seed: int = 123
    B: int = 16
    T: int = 32
    D: int = 128
    d_hidden: int = 128
    n_experts: int = 4
    k: int = 2
    capacity_factor: float = 5
    fast_implementation: bool = True
    match_routing_type: str = "topk"


def _timeit(callable, some_input):
    start = perf_counter()
    result = callable(some_input)
    return result, perf_counter()-start


def _print_diagnostics(d, l):
    for k, v in d.items():
        print(k, "\t", v)
    for k, v in l.items():
        print(k, "\t", v)


def _end_test():
    print("\n\n")


def _show_description():
    description = """Executes two tests comparing different Mixture-of-Experts (MoE) implementations for correctness and speed:
    -> [TEST 1]: _implementation_equivalence_test: 
    \t Compare outputs and speed of the vanilla and fast MoE implementations.
    -> [TEST 2]: _expert_equivalence_test: 
    \t Verify that MoE outputs match a base expert when all experts are identical to the base expert for both implemetations. Note: this returns True for ExpertChoiceRouting only when capacity is large enough. 
    """
    print(description)
    print("\n\n")


def _implementation_equivalence_test(config):
    print("[TEST 1]")
    torch.manual_seed(config.seed)
    routing_type = config.match_routing_type

    x = torch.randn(config.B, config.T, config.D)
    router_kwargs = dict(
        d_model=config.D,
        n_experts=config.n_experts,
        d_hidden=config.d_hidden,
        capacity_factor=config.capacity_factor 
    )
    if routing_type == "topk":
        router_kwargs["k"] = config.k

    moe_ref1 = TopKMoE if routing_type == "topk" else ExpertChoiceMoE
    moe_ref2 = TopKMoE_Fast if routing_type == "topk" else ExpertChoiceMoE_Fast

    moe1 = moe_ref1(**router_kwargs)
    moe2 = moe_ref2(**router_kwargs)
    
    # set weights in both implementations to be the same.
    # test only algorithmic difference.
    with torch.no_grad():
        moe2.gate.weight.copy_(moe1.gate.weight)
        moe2.experts = deepcopy(moe1.experts)
        
        result1, t1 = _timeit(moe1, x)
        result2, t2 = _timeit(moe2, x)

    # Exact verification
    torch.testing.assert_close(result1[0], result2[0], rtol=1e-5, atol=1e-6)
    logged_match = torch.allclose(result1[0], result2[0])

    print(f"Both implementations of `routing_type = {routing_type}` yield the same result: {logged_match}")
    print(f"Vanilla implementation took {t1} seconds.")
    print(f"Fast implementation took {t2} seconds.")
    print(f"Speedup: {round(t1/t2,2)} times.")
    print(" ")
    print("Diagnostics for slow implementation:")
    _print_diagnostics(result1[2], result1[1])
    print(" ")
    print("Diagnostics for fast implementation:")
    _print_diagnostics(result2[2], result2[1])

    _end_test()


def _expert_equivalence_test(config):
    print("[TEST 2]")
    torch.manual_seed(config.seed)
    use_fast_moe = config.fast_implementation

    x = torch.randn(config.B, config.T, config.D)
    base = FeedForwardExpert(config.D, config.d_hidden)

    topk_moe_ref = TopKMoE if not use_fast_moe else TopKMoE_Fast
    ec_moe_ref = ExpertChoiceMoE if not use_fast_moe else ExpertChoiceMoE_Fast

    topk = topk_moe_ref(config.D, config.n_experts, config.d_hidden, k=config.k, capacity_factor=config.capacity_factor)
    ec   = ec_moe_ref(config.D, config.n_experts, config.d_hidden, capacity_factor=config.capacity_factor)

    # Make all experts identical to `base`
    with torch.no_grad():
        for i in range(config.n_experts):
            for layer, base_layer in zip([topk.experts[i].fc1, topk.experts[i].fc2],
                                         [base.fc1, base.fc2]):
                layer.weight.copy_(base_layer.weight)
                layer.bias.copy_(base_layer.bias)
            for layer, base_layer in zip([ec.experts[i].fc1, ec.experts[i].fc2],
                                         [base.fc1, base.fc2]):
                layer.weight.copy_(base_layer.weight)
                layer.bias.copy_(base_layer.bias)

        # Uniform gating (all ones → softmax uniform). Ensures routing weights sum to 1
        # and with identical experts the MoE output matches the single-expert output.
        topk.gate.weight.fill_(1.0)
        ec.gate.weight.fill_(1.0)

    ref = base(x)
    y_topk, l_topk, d_topk = topk(x)
    y_ec,   l_ec,   d_ec   = ec(x)

    print("Using fast MoE implementation:", use_fast_moe, "\n")
    
    # check topk routing
    logged_topk = torch.allclose(y_topk, ref)
    print("TopKMoE equals single expert output", logged_topk)
    if not logged_topk:
        print("[TEST FAILED, IMPORTANT NOTICE] Implementation is suspected to be incorrect.")

    print("--"*30)
    print("TopKMoE diagnostics")
    _print_diagnostics(d_topk, l_topk)
    print(" ")

    # raise test-failed error after printing moe diagnostics 
    torch.testing.assert_close(y_topk, ref, rtol=1e-5, atol=1e-6)

    # check expert-choice routing
    logged_ec = torch.allclose(y_ec, ref)
    print("ExpertChoiceMoE equals single expert output", logged_ec)
    if not logged_ec:
        print(f"[TEST FAILED, IMPORTANT NOTICE] Check diagnostics for sufficient capacity before changing routing implementation.")
        print("ExpertChoiceMoE outputs match the single expert when no tokens are dropped.")

    print("--"*30)
    print("ExpertChoiceMoE diagnostics")
    _print_diagnostics(d_ec, l_ec)
    print(" ")
    
    # ec_moe will not match the base expert if capacity is low.
    # raise test-failed error after printing moe diagnostics to verify capacity (and dropped tokens). 
    torch.testing.assert_close(y_ec,   ref, rtol=1e-5, atol=1e-6)
    
    _end_test()


def main(config):
    _show_description()
    _implementation_equivalence_test(config)
    _expert_equivalence_test(config)


if __name__ == "__main__":
    config = tyro.cli(TestingConfig, default=vars(TestingConfig()))
    main(config)
