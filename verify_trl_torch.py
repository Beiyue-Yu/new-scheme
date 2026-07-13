"""
Verify the rewritten pure-torch TRL.forward matches tensorly's
tucker_to_tensor + tl_inner numerically, and that the full MSTR forward
+ backward works on GPU without any segfault / cuBLAS error.

Run inside the MSTR env:
    /home/wwj/anaconda3/envs/MSTR/bin/python verify_trl_torch.py
"""
import sys
import traceback

import torch
import numpy as np
import tensorly as tl


def banner(msg):
    print("\n" + "-" * 70)
    print(msg)
    print("-" * 70)


print("=" * 70)
print("ENV")
print("=" * 70)
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__, "cuda?", torch.cuda.is_available())
print("tensorly:", tl.__version__, "default backend:", tl.get_backend())


# ----------------------------------------------------------------------------
# 1. Numerical equivalence vs tensorly (numpy backend), on CPU.
# ----------------------------------------------------------------------------
banner("1. TRL.forward == tensorly reference (CPU)")

from src.model_improvements import TRL

torch.manual_seed(0)
configs = [
    dict(input_size=(1, 512, 1, 1), output_size=(1, 300), ranks=(400, 1, 1, 300)),
    dict(input_size=(1, 4096, 1, 1), output_size=(1, 300), ranks=(400, 1, 1, 300)),
]
B = 4

for cfg in configs:
    try:
        model = TRL(**cfg)
        model.eval()
        x = torch.randn(B, *cfg["input_size"][1:], requires_grad=True)
        with torch.no_grad():
            out_torch = model(x).clone()

        # reference using tensorly (numpy backend is fine on CPU)
        tl.set_backend("numpy")
        core = model.core.detach()
        factors = [f.detach() for f in model.factors]
        # tensorly wants numpy on numpy backend
        core_np = core.cpu().numpy()
        factors_np = [f.cpu().numpy() for f in factors]
        rw = tl.tucker_to_tensor((core_np, factors_np))
        x_np = x.detach().cpu().numpy()
        from tensorly.tenalg import inner as tl_inner
        from tensorly import ndim
        ref = tl_inner(x_np, rw, n_modes=ndim(x_np) - 1) + model.bias.detach().cpu().numpy()
        ref = torch.from_numpy(ref)

        max_diff = (out_torch - ref).abs().max().item()
        print(f"  {cfg['input_size']} ranks={cfg['ranks']}: "
              f"torch shape={tuple(out_torch.shape)} ref shape={tuple(ref.shape)} "
              f"max_diff={max_diff:.3e} "
              f"{'OK' if max_diff < 1e-4 else 'MISMATCH'}")
    except Exception:
        print(f"  {cfg} FAILED:")
        traceback.print_exc()


# ----------------------------------------------------------------------------
# 2. TRL forward+backward on GPU.
# ----------------------------------------------------------------------------
banner("2. TRL forward + backward on GPU")
if torch.cuda.is_available():
    for cfg in configs:
        try:
            model = TRL(**cfg).cuda()
            x = torch.randn(B, *cfg["input_size"][1:], device="cuda", requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            print(f"  {cfg['input_size']}: forward shape={tuple(out.shape)} "
                  f"is_cuda={out.is_cuda} backward OK, "
                  f"core.grad is_cuda={model.core.grad.is_cuda}")
        except Exception:
            print(f"  {cfg} FAILED:")
            traceback.print_exc()
else:
    print("  (cuda not available)")


# ----------------------------------------------------------------------------
# 3. Full MSTR forward + backward on GPU (dummy batch).
# ----------------------------------------------------------------------------
banner("3. Full MSTR forward + backward on GPU")
try:
    from src.utils_improvements import get_model_params
    from src.model_improvements import MSTR
    from spikingjelly.clock_driven import functional

    params = get_model_params(
        lr=0.001, first_additional_triplet=1, second_additional_triplet=1,
        reg_loss=True, additional_triplets_loss=True,
        dropout_encoder=0.3, dropout_decoder=0.15, additional_dropout=0.1,
        encoder_hidden_size=512, decoder_hidden_size=512,
        depth_transformer=1, momentum=0.1, snn_T=10, trl_rank=400,
    )
    for da, dv in [(512, 512), (512, 4096)]:
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            try:
                model = MSTR(params, input_size_audio=da, input_size_video=dv).to(device)
                model.train()
                bs = 8
                audio = torch.randn(bs, da, device=device)
                video = torch.randn(bs, 10, dv, device=device)
                neg_audio = torch.randn(bs, da, device=device)
                neg_video = torch.randn(bs, 10, dv, device=device)
                word_emb = torch.randn(bs, 300, device=device)
                neg_word_emb = torch.randn(bs, 300, device=device)
                # In MSTR, the video branch expects the temporal SNN to receive
                # per-timestep feature vectors (shape (B, D)), not the full
                # sequence. The temporal encoding is handled by _run_snn. We
                # therefore pass a single timestep's features as a representative
                # batch; the _encode_temporal_video loop repeats the SNN over T.
                video = video[:, 0, :]
                neg_video = neg_video[:, 0, :]
                out = model(audio, video, neg_audio, neg_video, word_emb, neg_word_emb)
                # MSTR.forward stores intermediates as attributes and returns None;
                # run the backward pass via model.backward(optimize) which computes
                # losses from the stored attributes, then step the optimizer.
                model.backward(optimize=True)
                functional.reset_net(model)
                print(f"  D_a={da} D_v={dv} device={device}: forward+backward OK "
                      f"(theta_a is_cuda={getattr(model.theta_a, 'is_cuda', 'NA')})")
                del model
                torch.cuda.empty_cache()
            except Exception:
                print(f"  D_a={da} D_v={dv} device={device} FAILED:")
                traceback.print_exc()
except Exception:
    print("MSTR test failed to run:")
    traceback.print_exc()


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
