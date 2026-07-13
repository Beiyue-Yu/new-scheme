#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import numpy as np
from einops import rearrange
# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math


torch.pi = math.pi

# user defined

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        # functional.reset_net(self.layers)
        return x

class TRL(nn.Module):
    """Tensor Regression Layer (Tucker-decomposition based linear regression).

    Original implementation relied on `tensorly.tucker_to_tensor` +
    `tensorly.tenalg.inner` for the forward pass. Under tensorly 0.7.0 the
    active backend is numpy (and `set_backend("pytorch")` does not reliably
    switch it in this env), so those calls round-trip CUDA nn.Parameters
    through numpy arrays. The resulting torch tensors are non-contiguous /
    C-subclassed, which cuBLAS rejects on Ada GPUs
    (CUBLAS_STATUS_NOT_SUPPORTED) — the Stage A segfault.

    `forward` and `penalty` are therefore reimplemented in pure torch,
    mathematically equivalent to the tensorly version for the configuration
    used by MSTR (input_size=(1, D, 1, 1), output_size=(1, 300)). The
    construction is general, though: it contracts x against the input
    factors and the core, then applies the output factors, exactly as a
    Tucker-regression contraction would.
    """
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__()
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(input_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)

        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)

        self.n_outputs = int(np.prod(self.output_size[1:]))

        # Core of the regression tensor weights (shape == ranks)
        self.core = nn.Parameter(torch.zeros(*self.ranks), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=True)

        # One factor per mode. The full Tucker weight tensor is built from
        # `core` and these factors; its modes are the input modes followed
        # by the output modes.
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])
        if len(weight_size) != len(self.ranks):
            raise ValueError(
                "TRL needs one rank per input/output mode: "
                f"got {len(self.ranks)} ranks for {len(weight_size)} modes"
            )
        if any(rank <= 0 for rank in self.ranks):
            raise ValueError(f"TRL ranks must be positive, got {self.ranks}")
        self.factors = []
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            f = nn.Parameter(torch.zeros(in_size, rank), requires_grad=True)
            self.factors.append(f)
            self.register_parameter('factor_{}'.format(index), f)

        # Init
        self.core.data.uniform_(-0.1, 0.1)
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)

        # Precompute layout metadata used by the pure-torch forward.
        n_input_modes = len(self.input_size) - 1   # exclude batch dim
        n_output_modes = len(self.output_size) - 1
        # input factor indices: 0 .. n_input_modes-1
        # output factor indices: n_input_modes .. n_input_modes+n_output_modes-1
        self._n_input_modes = n_input_modes
        self._n_output_modes = n_output_modes
        self._input_factor_idx = list(range(n_input_modes))
        self._output_factor_idx = list(range(n_input_modes,
                                              n_input_modes + n_output_modes))
        self._input_dims = self.input_size[1:]     # e.g. [D, 1, 1]
        self._output_dims = self.output_size[1:]   # e.g. [300]
        self._ranks = self.ranks                    # e.g. [R, 1, 1, 300]

    def forward(self, x):
        # x: (B, *input_size[1:])  e.g. (B, D, 1, 1)
        # Full Tucker-regression contraction: contract x with the input
        # factors, the core, and the output factors. Equivalent to
        #   W = tucker_to_tensor((core, factors))
        #   out = tl_inner(x, W, n_modes=len(input_modes)) + bias
        # but implemented in pure torch (no numpy round-trip), so it is safe
        # on GPU regardless of tensorly's active backend.
        expected = tuple(self.input_size[1:])
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"TRL expected input shaped (B, {', '.join(map(str, expected))}), "
                f"got {tuple(x.shape)}"
            )
        return self._forward_einsum(x, x.shape[0])

    def _forward_einsum(self, x, B):
        """Full Tucker-regression contraction via torch.einsum.

        Builds the equivalent of:
            W = tucker_to_tensor((core, factors))   # (1, *in, 1, *out) with batch-1 modes
            out = tl_inner(x, W, n_modes=len(input_modes))
        but entirely in torch on the original device/dtype.
        """
        # operand subscripts (1-indexed letters for clarity)
        # batch: b
        # input modes of x: i1, i2, ...  (size = self._input_dims)
        # output modes: o1, o2, ...      (size = self._output_dims)
        # core modes: r1, r2, ... rn     (size = self._ranks)
        import string
        letters = string.ascii_lowercase
        bi = iter(letters)
        b = next(bi)  # batch
        in_letters = [next(bi) for _ in self._input_dims]      # x's non-batch modes
        # core has len(self._ranks) modes; first n_input_modes are input ranks,
        # the rest are output ranks.
        rank_letters = [next(bi) for _ in self._ranks]
        out_letters = [next(bi) for _ in self._output_dims]    # output materialized modes

        n_in = self._n_input_modes
        in_rank_letters  = rank_letters[:n_in]
        out_rank_letters = rank_letters[n_in:]

        # x: (b, *in_letters)
        x_subs = b + ''.join(in_letters)
        # core: (*rank_letters)
        core_subs = ''.join(rank_letters)
        # input factor i: (in_letters[i], in_rank_letters[i])
        # output factor j: (out_letters[j], out_rank_letters[j])
        # output: (b, *out_letters)

        # Ensure x is contiguous. The model parameters (core, factors) are
        # already moved to the device by model.to(device); we must NOT call
        # .to() on them here because Parameter.to() returns a non-leaf tensor
        # that detaches from the autograd graph, which causes a device
        # mismatch ("expected cpu:0 but got cuda:0") in backward and segfaults
        # on Ada GPUs under torch 2.0.1+cu118.
        x = x.contiguous()
        operands = [x]
        subs = [x_subs]
        for i in self._input_factor_idx:
            operands.append(self.factors[i])
            subs.append(in_letters[i] + in_rank_letters[i])
        operands.append(self.core)
        subs.append(core_subs)
        for j, fidx in enumerate(self._output_factor_idx):
            operands.append(self.factors[fidx])
            subs.append(out_letters[j] + out_rank_letters[j])

        out_subs = b + ''.join(out_letters)
        eq = ','.join(subs) + '->' + out_subs
        result = torch.einsum(eq, *operands)

        # The MSTR config sets output_size=(1, 300): there is a leading
        # size-1 "batch-like" output mode that we drop, plus we add bias.
        # Reshape to (B, n_outputs) to match the original tl_inner result
        # shape (B, 300) and the downstream consumer.
        result = result.reshape(B, self.n_outputs)
        return result + self.bias

    def penalty(self, order=2):
        penalty = torch.norm(self.core, order)
        for f in self.factors:
            penalty = penalty + torch.norm(f, order)
        return penalty




class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,hidden_size=None):
        super(EmbeddingNet, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class SigmoidSpike(torch.autograd.Function):
    """Straight-through surrogate gradient for the LIF hard spike.

    The raw spike is ``(v >= v_threshold).float()`` — a non-differentiable
    comparison whose gradient is zero everywhere. Without a surrogate, no
    gradient ever reaches the SNN branch weights (confirmed: every SNNbranch
    parameter has ``grad=None`` after ``loss.backward()``), so the whole
    temporal pathway — half of the Tucker fusion input — never trains. That is
    a primary reason val HM stalls at ~3.4%.

    Forward  : returns the hard 0/1 spike (correct discrete dynamics).
    Backward : passes the incoming gradient through multiplied by the
               derivative of a sigmoid surrogate ``sigma(alpha*(v - v_th))``,
               which is largest near the threshold and decays away from it.
               This is the standard surrogate-gradient technique used by
               spikingjelly (``SigmoidGate``) and the STFT/SNN literature.

    NB: the threshold is saved as a plain Python float (NOT a CUDA tensor) to
    avoid the torch 2.0.1+cu118 ``d.is_cuda()`` CUDAGuardImpl assertion that
    fires after ~20k iterations when a saved CUDA tensor is involved.
    """
    @staticmethod
    def forward(ctx, v, v_threshold, alpha=4.0):
        threshold = (float(v_threshold.item()) if torch.is_tensor(v_threshold)
                     else float(v_threshold))
        spike = (v >= threshold).to(v.dtype)
        # Save only `v` as a tensor. The threshold is a scalar; saving it as a
        # CUDA buffer (which the DTH hook rewrites in-place every step) caused
        # two problems: (1) an in-place version mismatch, fixed by .clone(),
        # but (2) on torch 2.0.1+cu118 the cloned saved tensor occasionally
        # triggers `d.is_cuda() INTERNAL ASSERT FAILED` (CUDAGuardImpl) after
        # ~20k iterations — a known saved-tensor caching-allocator corruption
        # with custom autograd.Function on Ada GPUs. Saving the threshold as a
        # plain Python float sidesteps both issues entirely.
        ctx.save_for_backward(v)
        ctx.v_threshold = threshold
        ctx.alpha = alpha
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        (v,) = ctx.saved_tensors
        alpha = ctx.alpha
        v_threshold = ctx.v_threshold
        # derivative of sigmoid(alpha*(v - v_th))
        s = torch.sigmoid(alpha * (v - v_threshold))
        grad_v = grad_output * alpha * s * (1.0 - s)
        return grad_v, None, None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with a *dynamically adjustable* threshold.

    This is the STFT [Li et al., TIP 2024] spiking neuron. Unlike
    ``spikingjelly.clock_driven.neuron.IFNode`` (which the original MSTR used),
    this LIF

      * has a membrane-time constant ``tau`` so the potential *decays* between
        steps (Eq. 5 of the paper: ``tau * dV/dt = -V + R*I``);
      * keeps ``v`` and ``v_threshold`` as buffers, so the threshold can be
        rewritten in-place every time step by the dynamic-threshold hook
        (DTH, Eq. 9) and survives ``functional.reset_net``-style resets via
        :meth:`reset`;
      * uses a standard-operator straight-through surrogate gradient so the
        hard spike is differentiable for backprop — without it the entire SNN
        branch receives no gradient and never trains.

    The forward implements one discrete LIF step:

        v = v + (-(v - v_reset) + (R * I)) / tau     # integration (Eq. 5)
        spike = hard.detach() + soft - soft.detach()  # firing (surrogate grad)
        v = v * (1 - spike) + v_reset * spike         # reset on spike

    ``v_reset`` defaults to 0. ``R`` (membrane resistance) defaults to 1 so the
    input current equals the weighted input. ``surrogate_alpha`` controls the
    steepness of the surrogate sigmoid (larger = closer to the true derivative
    of a step, but more prone to vanishing gradients away from threshold).
    """

    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0, R: float = 1.0,
                 surrogate_alpha: float = 4.0):
        super().__init__()
        self.tau = float(tau)
        self.v_reset = float(v_reset)
        self.R = float(R)
        self.surrogate_alpha = float(surrogate_alpha)
        self.base_threshold = float(v_threshold)
        self.threshold_value = float(v_threshold)
        self.register_buffer('v_threshold', torch.tensor(float(v_threshold)))
        # `v` is per-sample runtime state (shape depends on batch size),
        # NOT a learned parameter — keep it out of state_dict so it never
        # causes a shape mismatch when loading a checkpoint trained with a
        # different batch size.
        self.v = None

    def reset(self):
        """Reset all runtime state between independent samples/batches."""
        self.v = None
        self.threshold_value = self.base_threshold
        self.v_threshold.fill_(self.base_threshold)

    def forward(self, I: torch.Tensor) -> torch.Tensor:
        # Lazy (re)initialise state to match the input shape/device.
        if (self.v is None or self.v.shape != I.shape or
                self.v.device != I.device or self.v.dtype != I.dtype):
            self.v = torch.zeros_like(I)
        # LIF integration (Eq. 5). v decays toward v_reset and integrates I.
        # Keep the graph across the T steps of one SNN call. _run_snn resets
        # state before and after every independent input, so this is bounded
        # BPTT rather than an unbounded graph across batches.
        v_new = self.v + (-(self.v - self.v_reset) + self.R * I) / self.tau
        # Standard-op STE avoids torch 2.0.1's long-run CUDA guard failure in
        # custom autograd.Function while preserving the exact hard forward and
        # sigmoid-surrogate backward.
        hard_spike = (v_new >= self.threshold_value).to(v_new.dtype)
        soft_spike = torch.sigmoid(self.surrogate_alpha * (v_new - self.threshold_value))
        spike = hard_spike.detach() + soft_spike - soft_spike.detach()
        # Hard reset: spiked neurons go back to v_reset. Keeping this update in
        # the graph lets later time steps supervise earlier membrane states.
        self.v = v_new * (1.0 - spike) + self.v_reset * spike
        return spike


class GlobalLocalPool(nn.Module):
    """Global-Local Pooling (GLP, Eq. 7 of STFT).

    Combines max pooling (global variation) and average pooling (local salient
    regions) into a single per-sample guidance scalar ``P_all`` and uses it to
    *gate* the SNN input current ``I``:

        P_all = 0.5 * (P_max + P_avg) + beta * P_max + (1 - beta) * P_avg
        I_hat = (1 + sigmoid(P_all)) * I

    where ``beta`` is a learnable scalar. ``P_max`` / ``P_avg`` are computed
    across the feature dimension so for a (B, D) input both are (B, 1) — a
    per-sample global energy estimate. This is the single-vector adaptation of
    the paper's pooling (which was written for conv feature maps); it keeps the
    same max/avg combination and learnable balance, but operates on vectors.

    The residual vector adaptation preserves signed SeLaVi features. Applying
    the paper's outer sigmoid directly would constrain all currents to [0, 1]
    before a neuron whose initial threshold is 1.
    """

    def __init__(self):
        super().__init__()
        # A logit keeps the max-vs-avg coefficient in the intended [0, 1]
        # range. logit=0 corresponds to the balanced beta=0.5 setting.
        self.beta_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit)

    def forward(self, I: torch.Tensor, return_context: bool = False):
        # I: (B, D). Pool over the feature dim -> per-sample scalars (B, 1).
        p_max = I.max(dim=-1, keepdim=True).values
        p_avg = I.mean(dim=-1, keepdim=True)
        p_all = 0.5 * (p_max + p_avg) + self.beta * p_max + (1.0 - self.beta) * p_avg
        # Eq. 7 applies a residual GLP modulation before the spiking neuron.
        # For pre-extracted signed vectors, applying the paper's final sigmoid
        # would destroy sign and bound every current below the initial LIF
        # threshold. Retain its residual term and use a bounded pool gate.
        refined = (1.0 + torch.sigmoid(p_all)) * I
        if return_context:
            return refined, p_all
        return refined


class STFTSNNBranch(nn.Module):
    """One STFT spatial-temporal SNN branch (audio *or* video).

    The STFT paper uses three Conv-SNN blocks. Since this repository consumes a
    single pre-extracted vector rather than feature maps, each convolution is
    adapted to Linear -> LayerNorm -> GLP -> LIF while retaining three blocks.

    Each ``step`` runs the three blocks and returns (B, output_size). The TSF in
    :meth:`MSTR._run_snn` collects these outputs and aggregates them.
    """

    def __init__(self, input_size, output_size, hidden_size,
                 tau: float = 2.0, v_threshold: float = 1.0, momentum: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Normalization before each LIF so the input current has enough dynamic
        # range to actually cross the firing threshold. Without it the Linear
        # outputs are mostly |I| < 1, and since the LIF steady-state is v = I
        # (tau-decay), v stays below v_threshold=1.0 and the neuron never
        # fires — collapsing the whole SNN branch output to zeros, which is
        # the root cause of the HM=0 regression.
        #
        # LayerNorm (NOT BatchNorm): the SNN is *stateful* — the LIF membrane
        # potential accumulates across the T time steps, so the distribution
        # feeding bn2 shifts every step. BatchNorm's running stats are a single
        # average over all steps and match NO individual step, which makes the
        # eval-mode output collapse to ~1/24 of its train-mode magnitude
        # (confirmed: spike rate 0.055 train -> 0.0005 eval, norm 2.35 -> 0.10).
        # That eval collapse is why val HM stalls at ~3.8% despite train loss
        # reaching 0.28. LayerNorm normalizes per-sample across features with
        # NO running stats, so it is identical in train and eval.
        self.ln1 = nn.LayerNorm(hidden_size)
        self.lif1 = LIFNeuron(tau=tau, v_threshold=v_threshold)
        self.glp1 = GlobalLocalPool()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.ln2 = nn.LayerNorm(output_size)
        self.lif2 = LIFNeuron(tau=tau, v_threshold=v_threshold)
        self.glp2 = GlobalLocalPool()
        # Both papers use three SNN blocks. The original adaptation only kept
        # two, which weakens temporal encoding and does not match Eq. 6.
        self.fc3 = nn.Linear(output_size, output_size)
        self.ln3 = nn.LayerNorm(output_size)
        self.lif3 = LIFNeuron(tau=tau, v_threshold=v_threshold)
        self.glp3 = GlobalLocalPool()

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()
        self.lif3.reset()

    def set_threshold(self, v_threshold: float):
        """Set all neurons' shared runtime threshold for the DTH hook."""
        value = float(v_threshold)
        for lif in (self.lif1, self.lif2, self.lif3):
            lif.threshold_value = value
            lif.v_threshold.fill_(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One LIF step. The GLP gates the input current before each neuron.
        I1 = self.fc1(x)
        I1 = self.ln1(I1)
        I1 = self.glp1(I1)
        s1 = self.lif1(I1)
        I2 = self.fc2(s1)
        I2 = self.ln2(I2)
        I2 = self.glp2(I2)
        s2 = self.lif2(I2)
        I3 = self.fc3(s2)
        I3 = self.ln3(I3)
        I3, p_all = self.glp3(I3, return_context=True)
        out = self.lif3(I3)
        self._last_p_all = p_all.detach()
        return out


class LatentKnowledgeCombiner(nn.Module):
    """Latent Knowledge Combiner (LKC, Eq. 1-4 of STFT).

    The inputs in this repository are one pre-extracted vector per modality,
    rather than frame sequences. Each knowledge slot is therefore a D x D
    transform. Eq. 1-3 are evaluated per sample, and the resulting audio,
    visual and shared latent knowledge are treated as three attention tokens.
    This is important: self-attention over the previous length-one sequence
    was only a learned linear transform, and its computed ``K_t`` never
    affected the returned features.
    """

    def __init__(self, dim: int = 300, n_slots: int = 4,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        # Each slot is a (D, D) latent knowledge matrix.
        self.slots = nn.ParameterList(
            [nn.Parameter(torch.empty(dim, dim)) for _ in range(n_slots)])
        for s in self.slots:
            nn.init.xavier_uniform_(s)
        # ReLU gate projection (Eq. 2).
        self.W_oa = nn.Linear(dim, dim)
        self.W_ov = nn.Linear(dim, dim)
        # Learnable update rate alpha (Eq. 3), sigmoid-bounded to (0, 1).
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.knowledge_prior = nn.Parameter(torch.zeros(dim))
        # Self-attention refinement (Eq. 4) over audio, video and K_t.
        self.sa = nn.MultiheadAttention(dim, num_heads=n_heads,
                                        dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _combine(self, X, modality: str):
        """Apply Eq. 1-2 to one modality. X: (B, D) -> K_o, P.

        ``modality`` is ``"a"`` or ``"v"`` and selects the per-modality ReLU
        gate projection (``W_oa`` / ``W_ov``) so both are actually exercised.
        """
        # K_oa = sum_i phi(K_i X) X   (Eq. 1). phi = sigmoid.
        K_o = torch.zeros_like(X)
        for s in self.slots:
            gate = torch.sigmoid(X @ s)            # (B, D)
            K_o = K_o + gate * X                    # (B, D)
        # P = ReLU(W_o K_o + b)  (Eq. 2) — per-modality projection.
        W_o = self.W_oa if modality == "a" else self.W_ov
        P = torch.relu(W_o(K_o))
        return K_o, P

    def forward(self, X_a, X_v):
        """Refine audio and visual features through shared latent knowledge.

        Inputs:  X_a, X_v : (B, D)
        Returns: R_a, R_v : (B, D)
        """
        K_oa, P_a = self._combine(X_a, modality="a")
        K_ov, P_v = self._combine(X_v, modality="v")

        # Vector form of Eq. 3. Elementwise products are the diagonal-vector
        # counterpart of P_a K_oa and P_v K_ov in the matrix formulation.
        prior = self.knowledge_prior.unsqueeze(0).expand_as(X_a)
        K_t = self.alpha * (P_a * K_oa + P_v * K_ov) + (1.0 - self.alpha) * prior

        # Eq. 4: reason jointly instead of applying degenerate length-one SA.
        K_toa = K_oa + P_a + K_t
        K_tov = K_ov + P_v + K_t
        tokens = torch.stack((K_toa, K_tov, K_t), dim=1)
        normed = self.attn_norm(tokens)
        attn_out, _ = self.sa(normed, normed, normed, need_weights=False)
        tokens = tokens + self.dropout(attn_out)
        tokens = tokens + self.dropout(self.mlp(self.ff_norm(tokens)))
        return tokens[:, 0, :], tokens[:, 1, :]


class TemporalSemanticTuckerFusion(nn.Module):
    """Hybrid STFT Tucker and restored MSTR spatial fusion.

    STFT first fuses each modality's temporal feature ``S`` with its semantic
    feature ``R`` through a low-rank Tucker core. The restored MSTR TRL
    feature is then fused into its own modality before shared audio-video
    reasoning.

    For a single vector input the bilinear model ``Y = T x_1 R x_2 S``
    (Eq. 10) would materialise a (D_s, D_t, K) tensor. Eq. 11-13 retain a
    three-mode low-rank core ``G[r_s, r_t, r_o]`` and an output factor. A
    two-dimensional core cannot represent this mapping and collapses Tucker
    fusion into a restricted multiplicative gate.
    """

    def __init__(self, dim: int = 300, rank: int = 60, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.rank = rank
        # Factor matrices for semantic (R) and temporal (S) per modality.
        self.U_sa = nn.Linear(dim, rank, bias=False)   # audio semantic factor
        self.U_ta = nn.Linear(dim, rank, bias=False)   # audio temporal factor
        self.U_sv = nn.Linear(dim, rank, bias=False)   # video semantic factor
        self.U_tv = nn.Linear(dim, rank, bias=False)   # video temporal factor
        # Three-mode Tucker cores and U^(k) output factors (Eq. 11-12).
        self.G_a = nn.Parameter(torch.empty(rank, rank, rank))
        self.G_v = nn.Parameter(torch.empty(rank, rank, rank))
        nn.init.normal_(self.G_a, mean=0.0, std=1.0 / rank)
        nn.init.normal_(self.G_v, mean=0.0, std=1.0 / rank)
        self.out_a = nn.Linear(rank, dim)
        self.out_v = nn.Linear(rank, dim)
        self.dropout = nn.Dropout(dropout)
        self.spatial_pos = nn.Parameter(torch.zeros(2, dim))
        nn.init.normal_(self.spatial_pos, mean=0.0, std=0.02)
        # Fuse each modality's temporal-semantic feature with its own TRL
        # spatial feature. Applying one module to a concatenated batch shares
        # the fusion weights between audio and video.
        self.spatial_fusion = Transformer(
            dim, depth=1, heads=8, dim_head=64, mlp_dim=dim * 2,
            dropout=dropout)
        # Joint audio-visual reasoning (STFT Eq. 14 / MSTR CMF).
        self.cross_attn = Transformer(dim, depth=1, heads=8,
                                      dim_head=64, mlp_dim=dim * 2,
                                      dropout=dropout)

    def _tucker(self, R, S, U_s, U_t, G, out_proj):
        """Compact Tucker bilinear fusion for one modality.

        R, S: (B, D). Returns (B, D).
        """
        r = U_s(R)                       # (B, rank)  == R^T U_s
        s = U_t(S)                       # (B, rank)  == S^T U_t
        # y[b,k] = sum_i sum_j r[b,i] * s[b,j] * G[i,j,k].
        mixed = torch.einsum('bi,bj,ijk->bk', r, s, G)
        return out_proj(self.dropout(mixed))

    def forward(self, R_a, S_a, R_v, S_v, P_a, P_v):
        """Fuse temporal, semantic and spatial features, then mix modalities.

        R_a, R_v: LKC-refined semantic features (B, D)
        S_a, S_v: SNN temporal features (B, D)
        P_a, P_v: MSTR TRL spatial features (B, D)
        Returns: Y_a, Y_v : (B, D)
        """
        Y_a = self._tucker(R_a, S_a, self.U_sa, self.U_ta, self.G_a, self.out_a)
        Y_v = self._tucker(R_v, S_v, self.U_sv, self.U_tv, self.G_v, self.out_v)

        batch_size = Y_a.shape[0]
        audio_tokens = torch.stack(
            (Y_a + self.spatial_pos[0], P_a + self.spatial_pos[1]), dim=1)
        video_tokens = torch.stack(
            (Y_v + self.spatial_pos[0], P_v + self.spatial_pos[1]), dim=1)
        spatial_tokens = self.spatial_fusion(
            torch.cat((audio_tokens, video_tokens), dim=0))
        Y_a = spatial_tokens[:batch_size, 0, :]
        Y_v = spatial_tokens[batch_size:, 0, :]

        # Joint reasoning with shared-weight cross-attention (Eq. 14).
        seq = torch.stack((Y_a, Y_v), dim=1)        # (B, 2, D)
        seq = self.cross_attn(seq)
        return seq[:, 0, :], seq[:, 1, :]


class MSTR(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(MSTR, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding
        self.dim_out = params_model['dim_out']
        # Number of classes
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.semantic_dim = params_model.get('stft_dim', 512)
        self.r_enc=params_model['dropout_encoder']#0.2 0.3
        self.r_proj=params_model['dropout_decoder']#0.3 0.1
        self.depth_transformer=params_model['depth_transformer']
        self.additional_triplets_loss=params_model['additional_triplets_loss']
        self.reg_loss=params_model['reg_loss']
        self.r_dec=params_model['additional_dropout']#0.5 0.15
        self.momentum=params_model['momentum']

        self.first_additional_triplet=params_model['first_additional_triplet']
        self.second_additional_triplet=params_model['second_additional_triplet']

        print('Initializing trainable models...', end='')

        self.A_enc = EmbeddingNet(
            input_size=input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=self.semantic_dim,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )
        self.V_enc = EmbeddingNet(
            input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=self.semantic_dim,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

        # Restore MSTR Eq. 4-7 spatial extraction. The original source encoded
        # batch size 256 in these tuples and reshaped every batch to it. TRL
        # treats the leading dimension as a placeholder, so rank 400 is kept
        # for 512-D SeLaVi inputs while runtime batch size remains dynamic.
        configured_trl_rank = params_model.get('trl_rank', 400)
        audio_trl_rank = min(configured_trl_rank, input_size_audio)
        video_trl_rank = min(configured_trl_rank, input_size_video)
        self.trl_a = TRL(
            ranks=(audio_trl_rank, 1, 1, self.semantic_dim),
            input_size=(1, input_size_audio, 1, 1),
            output_size=(1, self.semantic_dim))
        self.trl_v = TRL(
            ranks=(video_trl_rank, 1, 1, self.semantic_dim),
            input_size=(1, input_size_video, 1, 1),
            output_size=(1, self.semantic_dim))

        self.W_proj= EmbeddingNet(
            input_size=300,
            output_size=self.dim_out,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )

        self.D = EmbeddingNet(
            input_size=self.dim_out,
            output_size=300,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )


        # --- STFT spatial-temporal SNN branches ---
        # Replace the original Linear-IFNode-Linear-IFNode SNNBranch with the
        # STFT branch: Linear-LIF(GLP-guided)-Linear-LIF(GLP-guided), where
        # each LIF has a dynamic threshold (DTH) and the per-step outputs are
        # aggregated by a learned Time-Step Factor (TSF) instead of plain mean.
        snn_tau = params_model.get('snn_tau', 2.0)
        self.SNNbranchaudio = STFTSNNBranch(
            input_size=input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=self.semantic_dim,
            tau=snn_tau, v_threshold=1.0, momentum=self.momentum)
        self.SNNbranchvideo = STFTSNNBranch(
            input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=self.semantic_dim,
            tau=snn_tau, v_threshold=1.0, momentum=self.momentum)
        # Time-Step Factor: learns a per-step scalar weight for the TSF softmax.
        # Initialised so that softmax is near-uniform (all logits == 0).
        # NB: self.T is defined later (a few lines below); define it first.
        self.T = params_model['snn_T']
        self.tsf_logits = nn.Parameter(torch.zeros(self.T))

        # --- Latent Knowledge Combiner (LKC) ---
        # Explores latent cross-modal semantic relationships with `n_slots`
        # learnable knowledge slots, applied to each modality encoder output.
        n_slots = params_model.get('lkc_n_slots', 4)
        n_heads = params_model.get('lkc_n_heads', 8)
        if self.semantic_dim % n_heads != 0:
            raise ValueError(
                f"stft_dim={self.semantic_dim} must be divisible by "
                f"lkc_n_heads={n_heads}")
        self.lkc = LatentKnowledgeCombiner(
            dim=self.semantic_dim, n_slots=n_slots, n_heads=n_heads, dropout=self.r_enc)
        # --- Temporal-Semantic Tucker Fusion ---
        # Replaces the symmetric cross-attention fusion in _fuse_and_project.
        # Fuses temporal (SNN) and semantic (encoder) features per modality
        # with a low-rank Tucker core that preserves full second-order
        # interactions, then cross-modally mixes audio<->video.
        tucker_rank = params_model.get('tucker_rank', 60)
        self.tucker_fusion = TemporalSemanticTuckerFusion(
            dim=self.semantic_dim, rank=tucker_rank, dropout=self.r_enc)

        self.A_proj = EmbeddingNet(input_size=self.semantic_dim, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.V_proj = EmbeddingNet(input_size=self.semantic_dim, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.A_rec = EmbeddingNet(input_size=self.dim_out, output_size=self.semantic_dim, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        self.V_rec = EmbeddingNet(input_size=self.dim_out, output_size=self.semantic_dim, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']
        self.optimizer_gen = optim.Adam(list(self.A_proj.parameters()) + list(self.V_proj.parameters()) +
                                        list(self.A_rec.parameters()) + list(self.V_rec.parameters()) +
                                        list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
                                        list(self.D.parameters()) +
                                        list(self.W_proj.parameters())+list(self.SNNbranchaudio.parameters())+list(self.SNNbranchvideo.parameters())
                                        +list(self.lkc.parameters())+list(self.tucker_fusion.parameters())
                                        +list(self.trl_a.parameters())+list(self.trl_v.parameters())
                                        +[self.tsf_logits],
                                        lr=self.lr, weight_decay=1e-5, foreach=False)

        self.scheduler_gen =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        print('Done')

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    # ------------------------------------------------------------------
    # Modular building blocks shared by forward() and get_embeddings().
    # ------------------------------------------------------------------

    def _run_snn(self, snn_branch, x):
        """Run a STFT SNN branch for ``self.T`` time steps.

        Per-step outputs are aggregated with Eq. 8's sample-dependent temporal
        softmax. ``tsf_logits`` supplies only a global temporal prior; the
        observed spike tensor determines a separate weight vector per sample.

        The LIF neurons carry state across steps (membrane potential decay),
        so feeding the same input repeatedly is no longer a no-op: step ``t``
        starts from the decaying potential left by step ``t-1``.

        After each step the Dynamic Threshold Hook (DTH, Eq. 9) is applied: the
        LIF threshold for the next step is recomputed from the GLP pooling
        matrix and the entropy of the current step's output, suppressing spike
        noise when the output is information-rich.
        """
        # Every call represents an independent input. Resetting here prevents
        # positive samples from changing negative-sample membrane states and
        # prevents validation results from depending on data-loader order.
        snn_branch.reset()
        outs = []
        try:
            for _ in range(self.T):
                out = snn_branch(x)
                glp_p_all = getattr(snn_branch, '_last_p_all', None)
                if glp_p_all is not None:
                    self._dynamic_threshold(snn_branch, out, glp_p_all)
                outs.append(out)
            outs = torch.stack(outs, dim=1)  # (B, T, D)

            fused, weights = self._time_step_fusion(outs)
            snn_branch._last_tsf_weights = weights.detach()
            return fused
        finally:
            # Outputs retain the bounded T-step autograd graph; the module must
            # not retain it after returning.
            snn_branch.reset()

    def _time_step_fusion(self, outs):
        """Apply sample-dependent TSF to a (B, T, D) spike tensor.

        A feature-wise max saturates for a 512-D binary vector: one spike is
        enough to give almost every time step the same maximum. Total spike
        activity, scaled by sqrt(D), preserves temporal differences while
        keeping the score scale stable as the feature dimension changes.
        """
        if outs.ndim != 3 or outs.shape[1] != self.T:
            raise ValueError(
                f"Expected SNN outputs shaped (B, {self.T}, D), got {tuple(outs.shape)}"
            )
        feature_dim = outs.shape[-1]
        salience = outs.sum(dim=-1) / math.sqrt(feature_dim)
        weights = torch.softmax(
            salience + self.tsf_logits.view(1, self.T), dim=1
        )
        fused = (outs * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights

    @torch.no_grad()
    def _dynamic_threshold(self, snn_branch, out, glp_p_all):
        """Dynamic Threshold Hook (DTH, Eq. 9).

        Adjusts the LIF threshold based on the current SNN output's entropy
        and the GLP pooling matrix. If the entropy is high (rich information),
        the threshold is raised to suppress spike noise; if low, it is
        lowered. We use the per-sample normalised entropy of the output
        magnitude as the information measure.

        A binary spike vector is better modelled by Bernoulli entropy than by
        a softmax over feature positions. Sparse/constant outputs lower the
        threshold; information-rich spike patterns raise it. The update is
        smoothed and clamped because a scalar threshold is shared by the batch.
        """
        spike_rate = out.float().mean(dim=-1, keepdim=True).clamp(1e-6, 1.0 - 1e-6)
        entropy = -(spike_rate * torch.log(spike_rate) +
                    (1.0 - spike_rate) * torch.log(1.0 - spike_rate)) / math.log(2.0)
        pool_information = torch.sigmoid(glp_p_all)
        target = (0.5 * pool_information + entropy).clamp(0.25, 2.0)
        previous = snn_branch.lif3.threshold_value
        new_th = 0.5 * previous + 0.5 * target.mean().item()
        snn_branch.set_threshold(new_th)

    def _encode_temporal_audio(self, audio):
        """Encode the audio modality: A_enc (semantic) + STFT SNN (temporal).

        The semantic feature ``phi_a`` is refined by the LKC together with the
        video semantic feature (cross-modal latent alignment), then fused with
        the SNN temporal feature ``S_a`` through the Temporal-Semantic Tucker
        Fusion module. Returns (R_a, S_a): the LKC-refined semantic feature
        and the SNN temporal feature, both (B, semantic_dim)."""
        phi_a = self.A_enc(audio)
        S_a = self._run_snn(self.SNNbranchaudio, audio)
        return phi_a, S_a

    def _encode_temporal_video(self, video):
        """Encode the video modality: V_enc (semantic) + STFT SNN (temporal).
        See :meth:`_encode_temporal_audio`."""
        phi_v = self.V_enc(video)
        S_v = self._run_snn(self.SNNbranchvideo, video)
        return phi_v, S_v

    @staticmethod
    def _as_spatial_tensor(features):
        """Restore singleton tensor modes without assuming a batch size."""
        if features.ndim != 2:
            raise ValueError(
                f"Expected flattened features shaped (B, D), got {tuple(features.shape)}"
            )
        return features.unsqueeze(-1).unsqueeze(-1)

    def _encode_spatial(self, audio, video):
        """Encode MSTR spatial features with the audio/video TRL branches."""
        P_a = self.trl_a(self._as_spatial_tensor(audio))
        P_v = self.trl_v(self._as_spatial_tensor(video))
        return P_a, P_v

    def _fuse_and_project(self, phi_a, phi_v, S_a, S_v, P_a, P_v):
        """STFT second-stage fusion + projection.

        1. LKC refines the semantic features (phi_a, phi_v) via shared latent
           cross-modal knowledge slots -> (R_a, R_v).
        2. Temporal-Semantic Tucker Fusion fuses each (R, S) pair with a
           low-rank second-order core and cross-modally mixes them with a
           shared cross-attention -> (Y_a, Y_v).
        3. Project to the shared embedding space with V_proj / A_proj.
        Returns (theta_a, theta_v).
        """
        R_a, R_v = self.lkc(phi_a, phi_v)
        Y_a, Y_v = self.tucker_fusion(R_a, S_a, R_v, S_v, P_a, P_v)

        theta_v = self.V_proj(Y_v)
        theta_a = self.A_proj(Y_a)
        return theta_a, theta_v

    def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):
        # --- temporal-semantic branch (positive) ---
        # phi_* and phi_*1 are semantic_dim encoder and SNN features.
        self.phi_a, self.phi_a1 = self._encode_temporal_audio(audio)
        self.phi_v, self.phi_v1 = self._encode_temporal_video(image)
        # --- temporal-semantic branch (negative) ---
        self.phi_a_neg, self.phi_a_neg1 = self._encode_temporal_audio(negative_audio)
        self.phi_v_neg, self.phi_v_neg1 = self._encode_temporal_video(negative_image)
        self.phi_at, self.phi_vt = self._encode_spatial(audio, image)
        self.phi_at_neg, self.phi_vt_neg = self._encode_spatial(
            negative_audio, negative_image)
        # --- text / semantic projection ---
        self.w = word_embedding
        self.w_neg = negative_word_embedding
        self.theta_w = self.W_proj(word_embedding)
        self.theta_w_neg = self.W_proj(negative_word_embedding)
        self.rho_w = self.D(self.theta_w)

        # --- STFT second-stage fusion + projection ---
        # Fuses semantic (phi) + temporal (phi_*1) per modality via LKC + Tucker.
        self.theta_a, self.theta_v = self._fuse_and_project(
            self.phi_a, self.phi_v, self.phi_a1, self.phi_v1,
            self.phi_at, self.phi_vt)
        self.theta_a_neg, self.theta_v_neg = self._fuse_and_project(
            self.phi_a_neg, self.phi_v_neg, self.phi_a_neg1, self.phi_v_neg1,
            self.phi_at_neg, self.phi_vt_neg)

        # --- reconstruction heads ---
        self.phi_v_rec = self.V_rec(self.theta_v)
        self.phi_a_rec = self.A_rec(self.theta_a)

        self.rho_a = self.D(self.theta_a)
        self.rho_v = self.D(self.theta_v)

    def backward(self, optimize):
        # STFT Eq. 16: a joint audio-visual embedding is contrasted with its
        # positive text and both negative text / negative AV examples.
        theta_av = 0.5 * (self.theta_a + self.theta_v)
        theta_av_neg = 0.5 * (self.theta_a_neg + self.theta_v_neg)
        joint_triplet = 0.5 * (
            self.triplet_loss(theta_av, self.theta_w, self.theta_w_neg) +
            self.triplet_loss(self.theta_w, theta_av, theta_av_neg)
        )
        l_triplet = joint_triplet
        if self.additional_triplets_loss:
            # MSTR Eq. 13 supervises each modality independently.
            modality_triplet = 0.25 * (
                self.first_additional_triplet * (
                    self.triplet_loss(self.theta_a, self.theta_w, self.theta_w_neg) +
                    self.triplet_loss(self.theta_v, self.theta_w, self.theta_w_neg)) +
                self.second_additional_triplet * (
                    self.triplet_loss(self.theta_w, self.theta_a, self.theta_a_neg) +
                    self.triplet_loss(self.theta_w, self.theta_v, self.theta_v_neg)))
            l_triplet = 0.5 * (joint_triplet + modality_triplet)

        # Eq. 17 projection loss. Keeping both modality terms prevents one
        # branch from being hidden by a good average.
        l_projection = 0.5 * (
            self.criterion_reg(self.theta_a, self.theta_w) +
            self.criterion_reg(self.theta_v, self.theta_w)
        )

        # Eq. 18 reconstruction loss in both the encoder-feature and original
        # 300-D word spaces. Each term is averaged so its scale is independent
        # of how many reconstruction heads are present.
        l_reconstruction = (
            self.criterion_reg(self.phi_a_rec, self.phi_a) +
            self.criterion_reg(self.phi_v_rec, self.phi_v) +
            self.criterion_reg(self.rho_a, self.w) +
            self.criterion_reg(self.rho_v, self.w) +
            self.criterion_reg(self.rho_w, self.w)
        ) / 5.0

        # Paper setting: L_all = 0.5 L_t + 0.5 (L_p + L_r).
        loss_gen = 0.5 * l_triplet + 0.5 * (l_projection + l_reconstruction)

        if optimize == True:
            self.optimizer_gen.zero_grad()
            loss_gen.backward()
            self.optimizer_gen.step()

        loss = {'triplet': l_triplet, 'projection': l_projection,
                'reconstruction': l_reconstruction, 'gen': loss_gen}

        loss_numeric = loss_gen

        return loss_numeric, loss

    def optimize_params(self, audio, video, cls_numeric, cls_embedding,audio_negative, video_negative, negative_cls_embedding,optimize=False):

        self.forward(audio, video, audio_negative, video_negative, cls_embedding, negative_cls_embedding)

        loss_numeric, loss = self.backward(optimize)

        return loss_numeric, loss

    def get_embeddings(self, audio, video, embedding):
        # Inference path: only the positive branch is needed.
        # audio: (B, D_a), video: (B, D_v), embedding: (B, 300).

        phi_a, phi_a1 = self._encode_temporal_audio(audio)
        phi_v, phi_v1 = self._encode_temporal_video(video)
        phi_at, phi_vt = self._encode_spatial(audio, video)
        theta_w = self.W_proj(embedding)

        theta_a, theta_v = self._fuse_and_project(
            phi_a, phi_v, phi_a1, phi_v1, phi_at, phi_vt)
        return theta_a, theta_v, theta_w
