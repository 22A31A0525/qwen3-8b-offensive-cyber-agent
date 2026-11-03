"""Standalone fused-SDPA eligibility test. Run on a Kaggle Tesla T4.

Uses a single ~4096-token attention call, not the full 8B model.
This tests fused backend availability/gradients, NOT whole-model VRAM fit.
"""
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def run_trial(enable_gqa):
    device = "cuda"
    q = torch.randn((1, 32, 4096, 128), device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn((1, 8, 4096, 128), device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn((1, 8, 4096, 128), device=device, dtype=torch.float16, requires_grad=True)
    if not enable_gqa:
        k_use = k.repeat_interleave(4, dim=1)
        v_use = v.repeat_interleave(4, dim=1)
    else:
        k_use, v_use = k, v
    torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        result = F.scaled_dot_product_attention(
            q, k_use, v_use,
            is_causal=True, dropout_p=0.0,
            enable_gqa=enable_gqa,
        )
        result.square().mean().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[FUSED SDPA] mode={'native_gqa' if enable_gqa else 'expanded_kv'} PASS; peak={peak:.0f} MiB", flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required; run this file on a Kaggle GPU", file=sys.stderr)
        sys.exit(2)
    print(f"Torch={torch.__version__} GPU={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}",flush=True)
    print("MATH BACKEND IS EXCLUDED. Any failure explicitly reports fused path unavailable.",flush=True)
    outcomes=[]
    for mode in (True,False):
        try:
            run_trial(mode)
            outcomes.append((mode,True))
        except (RuntimeError,NotImplementedError) as exc:
            print(f"[FUSED SDPA] mode={'native_gqa' if mode else 'expanded_kv'} UNAVAILABLE: {type(exc).__name__}: {exc}",flush=True)
            outcomes.append((mode,False))
        finally:
            torch.cuda.empty_cache()
    if not any(ok for _,ok in outcomes):
        print("[FUSED SDPA] No tested efficient-only full-length path. Use the tiled fallback.",flush=True)
