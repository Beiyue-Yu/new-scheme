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


class CrossModalResidualGate(nn.Module):
    """Add sample-adaptive complementary information without erasing either modality.

    Cross-modal attention is useful when the modalities agree, but a direct
    fused representation can also amplify the modality-specific shortcut that
    causes the Seen bias in GZSL.  This block therefore predicts a bounded,
    feature-wise residual from the other modality and adds it to the original
    representation.  The source modality remains the identity path, while the
    gate can suppress contradictory complementary evidence per sample.
    """

    def __init__(self, dim: int, dropout: float = 0.1,
                 residual_scale: float = 0.2):
        super().__init__()
        if dim <= 0:
            raise ValueError("Cross-modal residual dimension must be positive")
        if residual_scale < 0:
            raise ValueError("Cross-modal residual scale must be non-negative")
        self.residual_scale = float(residual_scale)
        self.audio_from_video = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.video_from_audio = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.audio_gate = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.video_gate = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Sigmoid())

    def forward(self, audio: torch.Tensor, video: torch.Tensor):
        if audio.ndim != 2 or video.ndim != 2 or audio.shape != video.shape:
            raise ValueError(
                "Cross-modal residual expects audio/video tensors with the "
                f"same (B, D) shape, got {tuple(audio.shape)} and "
                f"{tuple(video.shape)}")
        audio_video = torch.cat((audio, video), dim=-1)
        video_audio = torch.cat((video, audio), dim=-1)
        audio_residual = F.layer_norm(
            self.audio_from_video(video), (audio.shape[-1],))
        video_residual = F.layer_norm(
            self.video_from_audio(audio), (video.shape[-1],))
        audio_gate = self.audio_gate(audio_video)
        video_gate = self.video_gate(video_audio)
        return (
            audio + self.residual_scale * audio_gate * audio_residual,
            video + self.residual_scale * video_gate * video_residual,
        )


class _GradientReverse(torch.autograd.Function):
    """Identity in the forward pass and a sign flip during backpropagation."""

    @staticmethod
    def forward(ctx, input_tensor, scale):
        ctx.scale = float(scale)
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.scale * grad_output, None


def gradient_reverse(input_tensor, scale=1.0):
    return _GradientReverse.apply(input_tensor, scale)


class VisualSemanticResidualDebiaser(nn.Module):
    """Split a fused visual feature into semantic and context-residual codes.

    This is a feature-level adaptation of semantic/residual decomposition for
    GZSL. It does not claim to recover raw-frame motion: the semantic code is
    supervised by the existing text objective, while the residual is retained
    only for reconstruction and adversarially discouraged from predicting text.
    """

    def __init__(self, dim, output_dim, dropout=0.1):
        super().__init__()
        if dim <= 0 or output_dim <= 0:
            raise ValueError("Debiaser dimensions must be positive")

        def encoder():
            return nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(dim, dim))

        self.semantic_encoder = encoder()
        self.residual_encoder = encoder()
        self.decoder = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim))
        # This probe learns to recover text-compatible labels from the
        # residual, while gradient reversal makes the residual encoder remove
        # that information instead of hiding semantic shortcuts there.
        self.residual_text_probe = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, output_dim))

    def forward(self, visual_feature):
        if visual_feature.ndim != 2:
            raise ValueError(
                "VisualSemanticResidualDebiaser expects a (B, D) tensor, got "
                f"{tuple(visual_feature.shape)}")
        semantic = self.semantic_encoder(visual_feature)
        residual = self.residual_encoder(visual_feature)
        reconstruction = self.decoder(torch.cat((semantic, residual), dim=-1))
        residual_text = self.residual_text_probe(gradient_reverse(residual))
        return semantic, residual, reconstruction, residual_text

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


class StableVectorTRL(nn.Module):
    """Stable low-rank TRL for already flattened modality features.

    The singleton spatial modes in the original four-factor TRL do not add
    expressive power: the contraction is a low-rank linear map.  Removing
    those modes avoids multiplying gradients through two extra tiny factors.
    """

    def __init__(self, input_size, output_size, rank):
        super().__init__()
        rank = min(int(rank), int(input_size), int(output_size))
        if rank <= 0:
            raise ValueError(f"StableVectorTRL rank must be positive, got {rank}")
        self.input_factor = nn.Linear(input_size, rank, bias=False)
        self.output_factor = nn.Linear(rank, output_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_size))
        nn.init.orthogonal_(self.input_factor.weight)
        nn.init.xavier_uniform_(self.output_factor.weight)

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError(
                "StableVectorTRL expected flattened features shaped (B, D), "
                f"got {tuple(x.shape)}")
        return self.output_factor(self.input_factor(x)) + self.bias


class SpatialReliabilityGate(nn.Module):
    """Estimate a bounded spatial-branch weight from two agreement statistics."""

    def __init__(self, initial_gate=0.25):
        super().__init__()
        if not 0.0 < initial_gate < 1.0:
            raise ValueError("initial_gate must be strictly between zero and one")
        self.projection = nn.Linear(2, 1)
        nn.init.zeros_(self.projection.weight)
        initial_logit = math.log(initial_gate / (1.0 - initial_gate))
        nn.init.constant_(self.projection.bias, initial_logit)

    def forward(self, semantic, spatial):
        if semantic.ndim != 2 or spatial.shape != semantic.shape:
            raise ValueError(
                "Spatial reliability expects aligned (B, D) semantic and "
                f"spatial features, got {tuple(semantic.shape)} and "
                f"{tuple(spatial.shape)}")
        cosine = F.cosine_similarity(semantic, spatial, dim=1, eps=1e-6)
        norm_ratio = torch.log(
            (spatial.norm(dim=1) + 1e-6) /
            (semantic.norm(dim=1) + 1e-6)).clamp(-5.0, 5.0)
        statistics = torch.stack((cosine, norm_ratio), dim=1)
        return torch.sigmoid(self.projection(statistics))




class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,
                 hidden_size=None, normalization="batchnorm"):
        super(EmbeddingNet, self).__init__()
        if normalization not in {"batchnorm", "layernorm"}:
            raise ValueError(
                "normalization must be 'batchnorm' or 'layernorm', got "
                f"{normalization!r}")

        def normalization_layer(features, batchnorm_momentum=None):
            if normalization == "layernorm":
                return nn.LayerNorm(features)
            if batchnorm_momentum is None:
                return nn.BatchNorm1d(num_features=features)
            return nn.BatchNorm1d(
                num_features=features, momentum=batchnorm_momentum)

        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(normalization_layer(hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(normalization_layer(output_size, momentum))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            # Preserve the historical one-layer EmbeddingNet behavior: its
            # output normalization was present even when ``use_bn`` was false.
            modules.append(normalization_layer(output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class RunningFeatureStandardizer(nn.Module):
    """Per-feature Z-score with batch updates and frozen inference statistics."""

    def __init__(self, feature_dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_var", torch.ones(feature_dim))
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward_group(self, *embeddings):
        if not embeddings:
            raise ValueError("At least one embedding tensor is required")
        joined = torch.cat(embeddings, dim=0)
        if self.training:
            mean = joined.mean(dim=0)
            variance = joined.var(dim=0, unbiased=False)
            with torch.no_grad():
                if self.num_batches_tracked.item() == 0:
                    self.running_mean.copy_(mean.detach())
                    self.running_var.copy_(variance.detach())
                else:
                    self.running_mean.lerp_(mean.detach(), self.momentum)
                    self.running_var.lerp_(variance.detach(), self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            mean = self.running_mean
            variance = self.running_var
        inverse_std = torch.rsqrt(variance.clamp_min(0.0) + self.eps)
        return tuple((embedding - mean) * inverse_std
                     for embedding in embeddings)

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
        # `Tensor.max(...).values` also saves a CUDA argmax index tensor for
        # backward. In torch 2.0.1+cu118 this path eventually trips an internal
        # CUDAGuard assertion in the long GLP + BPTT run on Ada GPUs. `amax`
        # computes the same forward value without the unused argmax output.
        p_max = torch.amax(I, dim=-1, keepdim=True)
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
                 tau: float = 2.0, v_threshold: float = 1.0,
                 momentum: float = 0.1, use_glp: bool = True,
                 membrane_readout_scale: float = 0.0):
        super().__init__()
        self.use_glp = bool(use_glp)
        self.membrane_readout_scale = float(membrane_readout_scale)
        if self.membrane_readout_scale < 0.0:
            raise ValueError("membrane_readout_scale must be non-negative")
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
        # Preserve the original checkpoint surface for the spike-only path.
        # The learned gate is present only in the explicit membrane-readout
        # ablation, where it bounds the continuous residual by the configured
        # maximum scale.
        self.membrane_readout_gate = (
            nn.Parameter(torch.tensor(0.0))
            if self.membrane_readout_scale > 0.0 else None)

    def reset(self):
        self.lif1.reset()
        self.lif2.reset()
        self.lif3.reset()

    def set_threshold(self, v_threshold):
        """Set the DTH threshold without coupling independent batch samples.

        The LIF comparison broadcasts a ``(B, 1)`` threshold over feature
        dimensions. Keeping that tensor as transient runtime state lets DTH
        adapt each sample independently; a batch-mean scalar made a sample's
        embedding change when unrelated samples were added to its batch.
        """
        if torch.is_tensor(v_threshold):
            value = v_threshold.detach()
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            if value.ndim != 2 or value.shape[1] != 1:
                raise ValueError(
                    "Per-sample DTH thresholds must have shape (B, 1), got "
                    f"{tuple(value.shape)}")
            if not torch.isfinite(value).all():
                raise ValueError("Per-sample DTH thresholds must be finite")
            for lif in (self.lif1, self.lif2, self.lif3):
                lif.threshold_value = value
            return

        value = float(v_threshold)
        for lif in (self.lif1, self.lif2, self.lif3):
            lif.threshold_value = value
            lif.v_threshold.fill_(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # One LIF step. The GLP gates the input current before each neuron.
        I1 = self.fc1(x)
        I1 = self.ln1(I1)
        if self.use_glp:
            I1 = self.glp1(I1)
        s1 = self.lif1(I1)
        I2 = self.fc2(s1)
        I2 = self.ln2(I2)
        if self.use_glp:
            I2 = self.glp2(I2)
        s2 = self.lif2(I2)
        I3 = self.fc3(s2)
        I3 = self.ln3(I3)
        if self.use_glp:
            I3, p_all = self.glp3(I3, return_context=True)
        else:
            p_all = I3.new_zeros(I3.shape[0], 1)
        out = self.lif3(I3)
        self._last_p_all = p_all.detach()
        self._last_spike_output = out
        if self.membrane_readout_gate is None:
            return out
        membrane = F.layer_norm(self.lif3.v, (self.lif3.v.shape[-1],))
        residual_scale = self.membrane_readout_scale * torch.sigmoid(
            self.membrane_readout_gate)
        return out + residual_scale * membrane


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

    def forward(self, R_a, S_a, R_v, S_v, P_a, P_v,
                spatial_reliability=None):
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
        spatial_a = spatial_tokens[:batch_size, 0, :]
        spatial_v = spatial_tokens[batch_size:, 0, :]
        if spatial_reliability is None:
            Y_a, Y_v = spatial_a, spatial_v
        else:
            gate_a, gate_v = spatial_reliability
            expected = (batch_size, 1)
            if tuple(gate_a.shape) != expected or tuple(gate_v.shape) != expected:
                raise ValueError(
                    "Spatial reliability gates must have shape "
                    f"{expected}, got {tuple(gate_a.shape)} and "
                    f"{tuple(gate_v.shape)}")
            # Gate the net spatial-fusion contribution. Zero bypasses the
            # branch, while one exactly recovers the ungated VectorTRL path.
            Y_a = Y_a + gate_a * (spatial_a - Y_a)
            Y_v = Y_v + gate_v * (spatial_v - Y_v)

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
        self.text_embedding_size = int(params_model.get('text_embedding_size', 300))
        if self.text_embedding_size <= 0:
            raise ValueError("text_embedding_size must be positive")
        self.r_enc=params_model['dropout_encoder']#0.2 0.3
        self.r_proj=params_model['dropout_decoder']#0.3 0.1
        self.depth_transformer=params_model['depth_transformer']
        self.additional_triplets_loss=params_model['additional_triplets_loss']
        self.reg_loss=params_model['reg_loss']
        self.r_dec=params_model['additional_dropout']#0.5 0.15
        self.momentum=params_model['momentum']

        self.first_additional_triplet=params_model['first_additional_triplet']
        self.second_additional_triplet=params_model['second_additional_triplet']
        self.use_glp = bool(params_model.get('use_glp', True))
        self.use_lkc = bool(params_model.get('use_lkc', True))
        self.snn_activity_floor_weight = float(
            params_model.get('snn_activity_floor_weight', 0.0))
        self.snn_min_spike_rate = float(
            params_model.get('snn_min_spike_rate', 0.05))
        self.snn_membrane_readout_scale = float(
            params_model.get('snn_membrane_readout_scale', 0.0))
        if self.snn_activity_floor_weight < 0.0:
            raise ValueError("snn_activity_floor_weight must be non-negative")
        if not 0.0 < self.snn_min_spike_rate < 1.0:
            raise ValueError("snn_min_spike_rate must be strictly between 0 and 1")
        if self.snn_membrane_readout_scale < 0.0:
            raise ValueError("snn_membrane_readout_scale must be non-negative")
        # Experimental-only switch for paired ablations against the historical
        # batch-shared DTH implementation. The default remains per-sample.
        self.legacy_batch_dth = bool(
            params_model.get('legacy_batch_dth', False))
        self.ahse_standardize = bool(
            params_model.get('ahse_standardize', False))
        self.semantic_geometry_weight = float(
            params_model.get('semantic_geometry_weight', 0.0))
        if self.semantic_geometry_weight < 0.0:
            raise ValueError("semantic_geometry_weight must be non-negative")
        self.semantic_contrastive_weight = float(
            params_model.get('semantic_contrastive_weight', 0.0))
        self.semantic_contrastive_temperature = float(
            params_model.get('semantic_contrastive_temperature', 0.1))
        if self.semantic_contrastive_weight < 0.0:
            raise ValueError("semantic_contrastive_weight must be non-negative")
        if self.semantic_contrastive_temperature <= 0.0:
            raise ValueError(
                "semantic_contrastive_temperature must be positive")
        self.pseudo_unseen_weight = float(
            params_model.get('pseudo_unseen_weight', 0.0))
        self.pseudo_unseen_temperature = float(
            params_model.get('pseudo_unseen_temperature', 0.15))
        self.pseudo_unseen_class_fraction = float(
            params_model.get('pseudo_unseen_class_fraction', 0.5))
        self.pseudo_unseen_min_classes = int(
            params_model.get('pseudo_unseen_min_classes', 2))
        if self.pseudo_unseen_weight < 0.0:
            raise ValueError("pseudo_unseen_weight must be non-negative")
        if self.pseudo_unseen_temperature <= 0.0:
            raise ValueError("pseudo_unseen_temperature must be positive")
        if not 0.0 < self.pseudo_unseen_class_fraction < 1.0:
            raise ValueError("pseudo_unseen_class_fraction must be in (0, 1)")
        if self.pseudo_unseen_min_classes < 2:
            raise ValueError("pseudo_unseen_min_classes must be at least 2")
        self.snn_temporal_consistency_weight = float(
            params_model.get('snn_temporal_consistency_weight', 0.0))
        self.snn_temporal_view_fraction = float(
            params_model.get('snn_temporal_view_fraction', 0.25))
        if self.snn_temporal_consistency_weight < 0.0:
            raise ValueError(
                "snn_temporal_consistency_weight must be non-negative")
        if not 0.0 < self.snn_temporal_view_fraction <= 1.0:
            raise ValueError(
                "snn_temporal_view_fraction must be in (0, 1]")
        self.temporal_quality_alignment_weight = float(
            params_model.get('temporal_quality_alignment_weight', 0.0))
        if self.temporal_quality_alignment_weight < 0.0:
            raise ValueError(
                "temporal_quality_alignment_weight must be non-negative")
        self.cross_modal_contrastive_weight = float(
            params_model.get('cross_modal_contrastive_weight', 0.0))
        self.cross_modal_contrastive_temperature = float(
            params_model.get('cross_modal_contrastive_temperature', 0.1))
        if self.cross_modal_contrastive_weight < 0.0:
            raise ValueError("cross_modal_contrastive_weight must be non-negative")
        if self.cross_modal_contrastive_temperature <= 0.0:
            raise ValueError(
                "cross_modal_contrastive_temperature must be positive")
        self.avla_contrastive_only = bool(
            params_model.get('avla_contrastive_only', False))
        self.avla_temperature = float(params_model.get('avla_temperature', 0.1))
        if self.avla_temperature <= 0.0:
            raise ValueError("avla_temperature must be positive")
        self.global_prototype_contrastive_weight = float(
            params_model.get('global_prototype_contrastive_weight', 0.0))
        self.global_prototype_contrastive_temperature = float(
            params_model.get('global_prototype_contrastive_temperature', 0.1))
        if self.global_prototype_contrastive_weight < 0.0:
            raise ValueError(
                "global_prototype_contrastive_weight must be non-negative")
        if self.global_prototype_contrastive_temperature <= 0.0:
            raise ValueError(
                "global_prototype_contrastive_temperature must be positive")
        if self.global_prototype_contrastive_weight > 0.0 and self.ahse_standardize:
            raise ValueError(
                "global prototype contrastive loss is incompatible with "
                "--ahse_standardize")
        self.semantic_hard_negative_weight = float(
            params_model.get('semantic_hard_negative_weight', 0.0))
        self.semantic_hard_negative_margin = float(
            params_model.get('semantic_hard_negative_margin', 0.1))
        if self.semantic_hard_negative_weight < 0.0:
            raise ValueError("semantic_hard_negative_weight must be non-negative")
        if self.semantic_hard_negative_margin < 0.0:
            raise ValueError("semantic_hard_negative_margin must be non-negative")
        self.semantic_batch_hard_weight = float(
            params_model.get('semantic_batch_hard_weight', 0.0))
        self.semantic_batch_hard_margin = float(
            params_model.get('semantic_batch_hard_margin', 0.1))
        self.semantic_batch_hard_neighbors = int(
            params_model.get('semantic_batch_hard_neighbors', 5))
        if self.semantic_batch_hard_weight < 0.0:
            raise ValueError("semantic_batch_hard_weight must be non-negative")
        if self.semantic_batch_hard_margin < 0.0:
            raise ValueError("semantic_batch_hard_margin must be non-negative")
        if self.semantic_batch_hard_neighbors <= 0:
            raise ValueError("semantic_batch_hard_neighbors must be positive")
        self.semantic_neighbor_rank_weight = float(
            params_model.get('semantic_neighbor_rank_weight', 0.0))
        self.semantic_neighbor_rank_margin = float(
            params_model.get('semantic_neighbor_rank_margin', 0.05))
        self.semantic_neighbor_rank_neighbors = int(
            params_model.get('semantic_neighbor_rank_neighbors', 5))
        if self.semantic_neighbor_rank_weight < 0.0:
            raise ValueError("semantic_neighbor_rank_weight must be non-negative")
        if self.semantic_neighbor_rank_margin < 0.0:
            raise ValueError("semantic_neighbor_rank_margin must be non-negative")
        if self.semantic_neighbor_rank_neighbors <= 0:
            raise ValueError("semantic_neighbor_rank_neighbors must be positive")
        self.semantic_mixup_weight = float(
            params_model.get('semantic_mixup_weight', 0.0))
        self.semantic_mixup_alpha = float(
            params_model.get('semantic_mixup_alpha', 1.0))
        if self.semantic_mixup_weight < 0.0:
            raise ValueError("semantic_mixup_weight must be non-negative")
        if self.semantic_mixup_alpha <= 0.0:
            raise ValueError("semantic_mixup_alpha must be positive")
        self.feature_mixup_weight = float(
            params_model.get('feature_mixup_weight', 0.0))
        self.feature_mixup_alpha = float(
            params_model.get('feature_mixup_alpha', 0.2))
        if self.feature_mixup_weight < 0.0:
            raise ValueError("feature_mixup_weight must be non-negative")
        if self.feature_mixup_alpha <= 0.0:
            raise ValueError("feature_mixup_alpha must be positive")
        self.feature_debias_weight = float(
            params_model.get('feature_debias_weight', 0.0))
        self.feature_debias_temperature = float(
            params_model.get('feature_debias_temperature', 0.1))
        if self.feature_debias_weight < 0.0:
            raise ValueError("feature_debias_weight must be non-negative")
        if self.feature_debias_temperature <= 0.0:
            raise ValueError("feature_debias_temperature must be positive")
        self.text_projection_norm = params_model.get(
            'text_projection_norm', 'batchnorm')
        self.use_stft_vector_trl = bool(
            params_model.get('stft_vector_trl', False))
        self.use_stft_spatial_reliability_gate = bool(
            params_model.get('stft_spatial_reliability_gate', False))
        if (self.use_stft_spatial_reliability_gate and
                not self.use_stft_vector_trl):
            raise ValueError(
                "stft_spatial_reliability_gate requires stft_vector_trl")
        self.lkc_residual_scale = float(params_model.get('lkc_residual_scale', 0.2))
        if self.lkc_residual_scale < 0.0:
            raise ValueError("lkc_residual_scale must be non-negative")
        self.use_cross_modal_residual = bool(
            params_model.get('cross_modal_residual', False))
        self.cross_modal_residual_scale = float(
            params_model.get('cross_modal_residual_scale', 0.2))
        if self.cross_modal_residual_scale < 0.0:
            raise ValueError("cross_modal_residual_scale must be non-negative")
        self.batch_labels = None
        self._feature_debias_state = None
        self._feature_mixup_inputs = None
        self._positive_snn_rates = None
        self._snn_temporal_view = None
        self._snn_temporal_teachers = None
        self._temporal_quality_features = None
        self._spatial_reliability_state = None
        self._snn_runtime_diagnostics = {}
        # Populated only for the opt-in training loss. Non-persistent buffers
        # preserve strict compatibility with all existing checkpoints.
        self.register_buffer(
            "_global_text_prototypes", torch.empty(0, self.text_embedding_size), persistent=False)
        self.register_buffer(
            "_global_prototype_class_ids", torch.empty(0, dtype=torch.long),
            persistent=False)
        self.register_buffer(
            "_pseudo_unseen_text_prototypes", torch.empty(0, self.text_embedding_size),
            persistent=False)
        self.register_buffer(
            "_pseudo_unseen_class_ids", torch.empty(0, dtype=torch.long),
            persistent=False)

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

        if self.use_stft_vector_trl:
            vector_rank = params_model.get('vector_trl_rank', 64)
            self.trl_a = StableVectorTRL(
                input_size_audio, self.semantic_dim, vector_rank)
            self.trl_v = StableVectorTRL(
                input_size_video, self.semantic_dim, vector_rank)
        else:
            # Original MSTR Eq. 4-7 spatial extraction. Keeping this exact
            # construction by default preserves all baseline checkpoints.
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
            input_size=self.text_embedding_size,
            output_size=self.dim_out,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True,
            normalization=self.text_projection_norm,
        )

        self.D = EmbeddingNet(
            input_size=self.dim_out,
            output_size=self.text_embedding_size,
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
            tau=snn_tau, v_threshold=1.0, momentum=self.momentum,
            use_glp=self.use_glp,
            membrane_readout_scale=self.snn_membrane_readout_scale)
        self.SNNbranchvideo = STFTSNNBranch(
            input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=self.semantic_dim,
            tau=snn_tau, v_threshold=1.0, momentum=self.momentum,
            use_glp=self.use_glp,
            membrane_readout_scale=self.snn_membrane_readout_scale)
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
        self.spatial_reliability_gate = (
            SpatialReliabilityGate(initial_gate=0.25)
            if self.use_stft_spatial_reliability_gate else None)
        # Optional feature-only context debiaser. It sits after SNN/Tucker
        # fusion, so it neither removes nor substitutes the existing SNN path.
        self.feature_debiaser = (
            VisualSemanticResidualDebiaser(
                self.semantic_dim, self.dim_out, dropout=self.r_enc)
            if self.feature_debias_weight > 0.0 else None)
        # Optional S-CMRL-inspired residual path. It is instantiated only for
        # the explicit ablation so legacy checkpoints retain an identical
        # state-dict and can still be loaded strictly.
        self.cross_modal_residual = (
            CrossModalResidualGate(
                self.semantic_dim, dropout=self.r_enc,
                residual_scale=self.cross_modal_residual_scale)
            if self.use_cross_modal_residual else None)

        self.A_proj = EmbeddingNet(input_size=self.semantic_dim, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.V_proj = EmbeddingNet(input_size=self.semantic_dim, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.A_rec = EmbeddingNet(input_size=self.dim_out, output_size=self.semantic_dim, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        self.V_rec = EmbeddingNet(input_size=self.dim_out, output_size=self.semantic_dim, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        if self.ahse_standardize:
            # MSTR evaluates audio and video separately, so both branches use
            # one audio-visual distribution while text keeps its own, as in
            # AHSE Stage I's modality-wise standardization.
            self.av_standardizer = RunningFeatureStandardizer(
                self.dim_out, momentum=self.momentum)
            self.text_standardizer = RunningFeatureStandardizer(
                self.dim_out, momentum=self.momentum)

        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']
        trainable_params = (
            list(self.A_proj.parameters()) + list(self.V_proj.parameters()) +
            list(self.A_rec.parameters()) + list(self.V_rec.parameters()) +
            list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
            list(self.D.parameters()) + list(self.W_proj.parameters()) +
            list(self.SNNbranchaudio.parameters()) +
            list(self.SNNbranchvideo.parameters()) + list(self.lkc.parameters()) +
            list(self.tucker_fusion.parameters()) + list(self.trl_a.parameters()) +
            list(self.trl_v.parameters()) + [self.tsf_logits]
        )
        if self.cross_modal_residual is not None:
            trainable_params += list(self.cross_modal_residual.parameters())
        if self.feature_debiaser is not None:
            trainable_params += list(self.feature_debiaser.parameters())
        if self.spatial_reliability_gate is not None:
            trainable_params += list(self.spatial_reliability_gate.parameters())
        self.optimizer_gen = optim.Adam(
            trainable_params, lr=self.lr, weight_decay=1e-5, foreach=False)

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
        spikes = []
        try:
            for _ in range(self.T):
                out = snn_branch(x)
                glp_p_all = getattr(snn_branch, '_last_p_all', None)
                if glp_p_all is not None:
                    self._dynamic_threshold(
                        snn_branch, snn_branch._last_spike_output, glp_p_all)
                outs.append(out)
                spikes.append(snn_branch._last_spike_output)
            outs = torch.stack(outs, dim=1)  # (B, T, D)
            spikes = torch.stack(spikes, dim=1)

            fused, weights = self._time_step_fusion(outs)
            snn_branch._last_tsf_weights = weights.detach()
            snn_branch._last_spike_rate = spikes.detach().mean()
            snn_branch._last_spike_rate_for_loss = spikes.mean()
            threshold = snn_branch.lif3.threshold_value
            snn_branch._last_dth_threshold = (
                threshold.detach().float().mean()
                if torch.is_tensor(threshold)
                else outs.new_tensor(float(threshold)))
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
        smoothed and clamped per sample, then broadcast over its feature
        dimension. It must not pool unrelated samples into one threshold.
        ``legacy_batch_dth`` is retained only for paired reproducibility
        experiments; it restores the historical batch-mean scalar update.
        """
        spike_rate = out.float().mean(dim=-1, keepdim=True).clamp(1e-6, 1.0 - 1e-6)
        entropy = -(spike_rate * torch.log(spike_rate) +
                    (1.0 - spike_rate) * torch.log(1.0 - spike_rate)) / math.log(2.0)
        pool_information = torch.sigmoid(glp_p_all)
        target = (0.5 * pool_information + entropy).clamp(0.25, 2.0)
        previous = snn_branch.lif3.threshold_value
        if self.legacy_batch_dth:
            if torch.is_tensor(previous):
                previous = previous.mean().item()
            new_th = 0.5 * float(previous) + 0.5 * target.mean().item()
            snn_branch.set_threshold(new_th)
            return
        if torch.is_tensor(previous):
            previous = previous.to(device=target.device, dtype=target.dtype)
        new_th = 0.5 * previous + 0.5 * target
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
        # Normalize spike activity before Tucker fusion so a modality with a
        # higher firing rate cannot overwhelm the semantic encoder.
        S_a = F.layer_norm(S_a, (self.semantic_dim,))
        return phi_a, S_a

    def _encode_temporal_video(self, video):
        """Encode the video modality: V_enc (semantic) + STFT SNN (temporal).
        See :meth:`_encode_temporal_audio`."""
        phi_v = self.V_enc(video)
        S_v = self._run_snn(self.SNNbranchvideo, video)
        S_v = F.layer_norm(S_v, (self.semantic_dim,))
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
        if self.use_stft_vector_trl:
            return self.trl_a(audio), self.trl_v(video)
        P_a = self.trl_a(self._as_spatial_tensor(audio))
        P_v = self.trl_v(self._as_spatial_tensor(video))
        return P_a, P_v

    def _apply_lkc_residual(self, phi_a, phi_v):
        """Apply LKC as a normalized, learnably gated residual."""
        if not self.use_lkc:
            return phi_a, phi_v
        refined_a, refined_v = self.lkc(phi_a, phi_v)
        gate = self.lkc_residual_scale * self.lkc.alpha
        residual_a = F.layer_norm(refined_a - phi_a, (self.semantic_dim,))
        residual_v = F.layer_norm(refined_v - phi_v, (self.semantic_dim,))
        return phi_a + gate * residual_a, phi_v + gate * residual_v

    def _fuse_and_project(self, phi_a, phi_v, S_a, S_v, P_a, P_v,
                          collect_feature_debias=False):
        """STFT second-stage fusion + projection.

        1. LKC refines the semantic features (phi_a, phi_v) via shared latent
           cross-modal knowledge slots -> (R_a, R_v).
        2. Temporal-Semantic Tucker Fusion fuses each (R, S) pair with a
           low-rank second-order core and cross-modally mixes them with a
           shared cross-attention -> (Y_a, Y_v).
        3. Project to the shared embedding space with V_proj / A_proj.
        Returns (theta_a, theta_v).
        """
        R_a, R_v = self._apply_lkc_residual(phi_a, phi_v)
        spatial_reliability = None
        if self.spatial_reliability_gate is not None:
            gate_a = self.spatial_reliability_gate(R_a, P_a)
            gate_v = self.spatial_reliability_gate(R_v, P_v)
            spatial_reliability = (gate_a, gate_v)
            if collect_feature_debias:
                self._spatial_reliability_state = (
                    gate_a.detach(), gate_v.detach())
        Y_a, Y_v = self.tucker_fusion(
            R_a, S_a, R_v, S_v, P_a, P_v,
            spatial_reliability=spatial_reliability)
        if self.cross_modal_residual is not None:
            Y_a, Y_v = self.cross_modal_residual(Y_a, Y_v)

        if self.feature_debiaser is not None:
            # Tucker fusion is intentionally unconstrained in scale.  The
            # debiaser is an auxiliary representation regularizer, so feeding
            # its decoder that raw magnitude would make reconstruction dwarf
            # the task loss.  Decompose a unit-scale visual direction instead.
            normalized_visual = F.layer_norm(Y_v, (self.semantic_dim,))
            semantic_visual, residual_visual, reconstructed_visual, residual_text = (
                self.feature_debiaser(normalized_visual))
            if collect_feature_debias:
                self._feature_debias_state = {
                    "source": normalized_visual.detach(),
                    "semantic": semantic_visual,
                    "residual": residual_visual,
                    "reconstruction": reconstructed_visual,
                    "residual_text": residual_text,
                }
            Y_v = semantic_visual

        theta_v = self.V_proj(Y_v)
        theta_a = self.A_proj(Y_a)
        return theta_a, theta_v

    def _standardize_training_embeddings(
            self, theta_a, theta_v, theta_a_neg, theta_v_neg,
            theta_w, theta_w_neg):
        if not self.ahse_standardize:
            return (theta_a, theta_v, theta_a_neg, theta_v_neg,
                    theta_w, theta_w_neg)
        theta_a, theta_v, theta_a_neg, theta_v_neg = (
            self.av_standardizer.forward_group(
                theta_a, theta_v, theta_a_neg, theta_v_neg))
        theta_w, theta_w_neg = self.text_standardizer.forward_group(
            theta_w, theta_w_neg)
        return (theta_a, theta_v, theta_a_neg, theta_v_neg,
                theta_w, theta_w_neg)

    def _standardize_inference_embeddings(self, theta_a, theta_v, theta_w):
        if not self.ahse_standardize:
            return theta_a, theta_v, theta_w
        theta_a, theta_v = self.av_standardizer.forward_group(theta_a, theta_v)
        theta_w, = self.text_standardizer.forward_group(theta_w)
        return theta_a, theta_v, theta_w

    def _semantic_geometry_loss(self):
        """Preserve pairwise text geometry through the learned projection."""
        projected = torch.cat((self.theta_w_raw, self.theta_w_neg_raw), dim=0)
        original = torch.cat((self.w, self.w_neg), dim=0).detach()
        projected = F.normalize(projected, dim=1)
        original = F.normalize(original, dim=1)
        projected_similarity = projected @ projected.transpose(0, 1)
        original_similarity = original @ original.transpose(0, 1)
        off_diagonal = ~torch.eye(
            projected_similarity.shape[0], dtype=torch.bool,
            device=projected_similarity.device)
        return F.smooth_l1_loss(
            projected_similarity[off_diagonal],
            original_similarity[off_diagonal])

    def _semantic_contrastive_loss(self):
        """Align audio, video, fused AV, and text at the class level.

        The existing triplet objective sees one random negative at a time. This
        supervised multi-positive InfoNCE term uses every other Seen class in
        the class-balanced batch as a semantic negative, while repeated samples
        of the same class remain positives.  The fused AV query is included so
        training directly supervises the representation used by combined
        evaluation.  It therefore strengthens the audio/video/text decision
        geometry without accessing held-out audio/video examples.
        """
        labels = self.batch_labels
        if labels is None:
            raise RuntimeError(
                "semantic contrastive loss requires batch class labels")
        labels = labels.reshape(-1)
        if labels.shape[0] != self.theta_a.shape[0]:
            raise ValueError(
                "semantic contrastive labels must match the batch size")

        positive_mask = labels[:, None].eq(labels[None, :])

        def multi_positive_nce(queries, keys):
            logits = F.normalize(queries, dim=1) @ F.normalize(
                keys, dim=1).transpose(0, 1)
            logits = logits / self.semantic_contrastive_temperature
            all_log_prob = torch.logsumexp(logits, dim=1)
            positive_log_prob = torch.logsumexp(
                logits.masked_fill(~positive_mask, -torch.inf), dim=1)
            return (all_log_prob - positive_log_prob).mean()

        audio_to_text = multi_positive_nce(self.theta_a, self.theta_w)
        video_to_text = multi_positive_nce(self.theta_v, self.theta_w)
        fused_av = 0.5 * (self.theta_a + self.theta_v)
        av_to_text = multi_positive_nce(fused_av, self.theta_w)
        text_to_audio = multi_positive_nce(self.theta_w, self.theta_a)
        text_to_video = multi_positive_nce(self.theta_w, self.theta_v)
        text_to_av = multi_positive_nce(self.theta_w, fused_av)
        return (
            audio_to_text + video_to_text + av_to_text + text_to_audio +
            text_to_video + text_to_av
        ) / 6.0

    def set_pseudo_unseen_text_prototypes(self, text_prototypes, class_ids):
        """Set the train-only class dictionary used by episodic training.

        The dictionary contains only classes available to the current training
        stage. It is a non-persistent buffer so enabling this auxiliary loss
        does not change checkpoint compatibility or inference behavior.
        """
        text_prototypes = torch.as_tensor(text_prototypes, dtype=torch.float32)
        class_ids = torch.as_tensor(class_ids, dtype=torch.long).reshape(-1)
        if (text_prototypes.ndim != 2 or
                text_prototypes.shape[1] != self.text_embedding_size):
            raise ValueError(
                "pseudo-Unseen text prototypes must have shape "
                f"(classes, {self.text_embedding_size})")
        if text_prototypes.shape[0] != class_ids.numel() or class_ids.numel() < 2:
            raise ValueError(
                "pseudo-Unseen text prototypes and class ids must be aligned")
        if class_ids.unique().numel() != class_ids.numel():
            raise ValueError("pseudo-Unseen class ids must be unique")
        self._pseudo_unseen_text_prototypes = text_prototypes.detach().clone()
        self._pseudo_unseen_class_ids = class_ids.detach().clone()

    def _pseudo_unseen_episode_loss(self):
        """Train-only episodic transfer loss over disjoint class subsets.

        A random subset of the classes present in the batch is treated as
        pseudo-Unseen query classes. Query class prototypes are formed from
        their current AV embeddings, but classification uses the complete
        train-stage text dictionary, including the held-out support classes.
        No validation/test audio or video is accessed.
        """
        labels = self.batch_labels
        if labels is None:
            raise RuntimeError(
                "pseudo-Unseen episodic loss requires batch class labels")
        labels = labels.reshape(-1)
        if labels.shape[0] != self.theta_a.shape[0]:
            raise ValueError(
                "pseudo-Unseen labels must match the batch size")
        if self._pseudo_unseen_class_ids.numel() < 2:
            raise RuntimeError(
                "set_pseudo_unseen_text_prototypes must be called before "
                "enabling the pseudo-Unseen loss")

        batch_classes = torch.unique(labels, sorted=True)
        if batch_classes.numel() < self.pseudo_unseen_min_classes + 1:
            return self.theta_a.new_zeros(())
        query_count = int(round(
            batch_classes.numel() * self.pseudo_unseen_class_fraction))
        query_count = max(self.pseudo_unseen_min_classes, query_count)
        query_count = min(query_count, batch_classes.numel() - 1)
        if query_count < self.pseudo_unseen_min_classes:
            return self.theta_a.new_zeros(())

        permutation = torch.randperm(batch_classes.numel(), device=labels.device)
        query_classes = batch_classes[permutation[:query_count]]
        query_mask = torch.isin(labels, query_classes)
        query_labels = labels[query_mask]
        query_class_ids, query_inverse = torch.unique(
            query_labels, sorted=True, return_inverse=True)

        class_count = query_class_ids.numel()
        if class_count < self.pseudo_unseen_min_classes:
            return self.theta_a.new_zeros(())

        def class_average(embeddings):
            prototypes = embeddings.new_zeros(class_count, embeddings.shape[1])
            prototypes.index_add_(0, query_inverse, embeddings[query_mask])
            counts = torch.bincount(
                query_inverse, minlength=class_count).to(
                    dtype=embeddings.dtype, device=embeddings.device)
            return prototypes / counts[:, None]

        query_audio = class_average(self.theta_a)
        query_video = class_average(self.theta_v)
        query_joint = 0.5 * (query_audio + query_video)

        episode_class_ids = self._pseudo_unseen_class_ids.to(labels.device)
        matches = query_class_ids[:, None].eq(episode_class_ids[None, :])
        if not matches.any(dim=1).all():
            missing = query_class_ids[~matches.any(dim=1)].tolist()
            raise ValueError(
                "pseudo-Unseen query classes missing from train dictionary: "
                f"{missing}")
        targets = matches.to(dtype=torch.long).argmax(dim=1)
        text = F.normalize(
            self.W_proj(self._pseudo_unseen_text_prototypes.to(labels.device)),
            dim=1)

        def classify(embeddings):
            logits = F.normalize(embeddings, dim=1) @ text.transpose(0, 1)
            return F.cross_entropy(
                logits / self.pseudo_unseen_temperature, targets)

        return (classify(query_audio) + classify(query_video) +
                classify(query_joint)) / 3.0

    def _cross_modal_contrastive_loss(self):
        """Align class-level audio and video semantics in the shared space.

        MSTR already supervises both modalities against text, but it has no
        direct audio-video consistency objective.  This symmetric
        multi-positive InfoNCE term treats every same-class item as a positive
        so it aligns event semantics rather than merely matching a clip's
        incidental audio-video background.  It is auxiliary to the original
        triplet, projection, and reconstruction losses.
        """
        labels = self.batch_labels
        if labels is None:
            raise RuntimeError(
                "cross-modal contrastive loss requires batch class labels")
        labels = labels.reshape(-1)
        if labels.shape[0] != self.theta_a.shape[0]:
            raise ValueError(
                "cross-modal contrastive labels must match the batch size")

        positive_mask = labels[:, None].eq(labels[None, :])

        def multi_positive_nce(queries, keys):
            logits = F.normalize(queries, dim=1) @ F.normalize(
                keys, dim=1).transpose(0, 1)
            logits = logits / self.cross_modal_contrastive_temperature
            return (
                torch.logsumexp(logits, dim=1) - torch.logsumexp(
                    logits.masked_fill(~positive_mask, -torch.inf), dim=1)
            ).mean()

        return 0.5 * (
            multi_positive_nce(self.theta_a, self.theta_v) +
            multi_positive_nce(self.theta_v, self.theta_a))

    def _avla_contrastive_loss(self):
        """Standalone joint AV-language alignment over class-level prototypes.

        This follows the supervision form used by EZ-AVGZL while preserving the
        complete STFT SNN/LKC/Tucker audio-video encoder.  The original MSTR
        reconstruction and triplet terms are deliberately not combined with
        this loss, making it a controlled objective replacement rather than
        another weak auxiliary penalty.
        """
        labels = self.batch_labels
        if labels is None:
            raise RuntimeError("AV-language alignment requires batch class labels")
        labels = labels.reshape(-1)
        if labels.shape[0] != self.theta_w.shape[0]:
            raise ValueError("AV-language labels must match the batch size")

        _, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        class_count = int(inverse.max().item()) + 1
        if class_count < 2:
            raise ValueError("AV-language alignment requires at least two classes")
        class_text = self.theta_w.new_zeros(class_count, self.theta_w.shape[1])
        class_text.index_add_(0, inverse, self.theta_w)
        class_counts = torch.bincount(
            inverse, minlength=class_count).to(
                dtype=self.theta_w.dtype, device=self.theta_w.device)
        class_text = class_text / class_counts[:, None]

        theta_av = 0.5 * (self.theta_a + self.theta_v)
        logits = F.normalize(theta_av, dim=1) @ F.normalize(
            class_text, dim=1).transpose(0, 1)
        return F.cross_entropy(logits / self.avla_temperature, inverse)

    def set_global_text_prototypes(self, text_prototypes, class_ids):
        """Set the final task's semantic dictionary for the training loss.

        Generalized ZSL permits final unseen-class semantic prototypes. No
        held-out audio/video example is used here. The buffers are deliberately
        non-persistent, because inference does not require this loss.
        """
        text_prototypes = torch.as_tensor(text_prototypes, dtype=torch.float32)
        class_ids = torch.as_tensor(class_ids, dtype=torch.long).reshape(-1)
        if (text_prototypes.ndim != 2 or
                text_prototypes.shape[1] != self.text_embedding_size):
            raise ValueError(
                "global text prototypes must have shape "
                f"(classes, {self.text_embedding_size})")
        if text_prototypes.shape[0] != class_ids.numel() or class_ids.numel() == 0:
            raise ValueError(
                "global prototype texts and class ids must be non-empty and aligned")
        if class_ids.unique().numel() != class_ids.numel():
            raise ValueError("global prototype class ids must be unique")
        self._global_text_prototypes = text_prototypes.detach().clone()
        self._global_prototype_class_ids = class_ids.detach().clone()

    def _global_prototype_contrastive_loss(self):
        """Contrast training AV embeddings with the full final class dictionary."""
        if self._global_text_prototypes.numel() == 0:
            raise RuntimeError(
                "global prototype contrastive loss requires final task text "
                "prototypes; call set_global_text_prototypes before training")
        if self.batch_labels is None:
            raise RuntimeError("global prototype contrastive loss requires batch labels")

        labels = self.batch_labels.reshape(-1).to(
            device=self._global_prototype_class_ids.device, dtype=torch.long)
        matches = labels[:, None].eq(self._global_prototype_class_ids[None, :])
        if not matches.any(dim=1).all():
            missing = labels[~matches.any(dim=1)].unique().tolist()
            raise ValueError(
                "training labels missing from global prototype dictionary: "
                f"{missing}")
        targets = matches.to(dtype=torch.long).argmax(dim=1)
        prototype_embeddings = F.normalize(
            self.W_proj(self._global_text_prototypes), dim=1)

        def prototype_nce(embeddings):
            logits = F.normalize(embeddings, dim=1) @ prototype_embeddings.T
            return F.cross_entropy(
                logits / self.global_prototype_contrastive_temperature, targets)

        theta_av = 0.5 * (self.theta_a + self.theta_v)
        return (prototype_nce(self.theta_a) + prototype_nce(self.theta_v) +
                prototype_nce(theta_av)) / 3.0

    def _semantic_hard_negative_loss(self):
        """Separate each modality from its closest different Seen-class text.

        Random triplets often sample an easy negative.  This term chooses the
        most semantically similar *different-label* Word2Vec item present in
        the current class-balanced batch, then enforces a cosine margin in the
        learned shared space.  Raw text chooses the hard class; projected text
        remains trainable through the ranking loss.
        """
        labels = self.batch_labels
        if labels is None:
            raise RuntimeError(
                "semantic hard-negative loss requires batch class labels")
        labels = labels.reshape(-1)
        if labels.shape[0] != self.theta_a.shape[0]:
            raise ValueError(
                "semantic hard-negative labels must match the batch size")

        different_class = labels[:, None].ne(labels[None, :])
        valid_rows = different_class.any(dim=1)
        if not valid_rows.any():
            return self.theta_a.new_zeros(())

        raw_text = F.normalize(self.w.detach(), dim=1)
        semantic_similarity = raw_text @ raw_text.transpose(0, 1)
        semantic_similarity = semantic_similarity.masked_fill(
            ~different_class, -torch.inf)
        hard_indices = semantic_similarity.argmax(dim=1)
        positive_text = F.normalize(self.theta_w, dim=1)
        negative_text = positive_text[hard_indices]

        def ranking_loss(modality):
            modality = F.normalize(modality, dim=1)
            positive_score = (modality * positive_text).sum(dim=1)
            negative_score = (modality * negative_text).sum(dim=1)
            return F.relu(
                negative_score - positive_score +
                self.semantic_hard_negative_margin)[valid_rows].mean()

        return 0.5 * (ranking_loss(self.theta_a) + ranking_loss(self.theta_v))

    def _semantic_batch_hard_loss(self):
        """Separate each AV embedding from its currently closest semantic peer.

        MSTR's sampled triplets use one arbitrary negative clip.  Here the
        raw word vectors first limit candidates to a small, semantically
        related neighbourhood; the learned AV-to-text scores then select the
        currently most confusable class prototype.  Class-level aggregation
        removes duplicate examples from the decision, and uses Seen training
        classes only.
        """
        labels = self.batch_labels
        if labels is None:
            raise RuntimeError("semantic batch-hard loss requires batch class labels")
        labels = labels.reshape(-1)
        if labels.shape[0] != self.theta_a.shape[0]:
            raise ValueError("semantic batch-hard labels must match batch size")

        class_ids, inverse = torch.unique(labels, sorted=True,
                                          return_inverse=True)
        class_count = int(class_ids.numel())
        if class_count < 2:
            return self.theta_a.new_zeros(())

        def class_average(embeddings):
            prototypes = embeddings.new_zeros(
                class_count, embeddings.shape[1])
            prototypes.index_add_(0, inverse, embeddings)
            counts = torch.bincount(inverse, minlength=class_count).to(
                dtype=embeddings.dtype, device=embeddings.device)
            return prototypes / counts[:, None]

        raw_text = F.normalize(class_average(self.w.detach()), dim=1)
        text_prototypes = F.normalize(class_average(self.theta_w), dim=1)
        semantic_similarity = raw_text @ raw_text.transpose(0, 1)
        semantic_similarity.fill_diagonal_(-torch.inf)
        neighbor_count = min(self.semantic_batch_hard_neighbors,
                             class_count - 1)
        semantic_neighbors = semantic_similarity.topk(
            neighbor_count, dim=1).indices

        theta_av = F.normalize(0.5 * (self.theta_a + self.theta_v), dim=1)
        class_scores = theta_av @ text_prototypes.transpose(0, 1)
        positive_scores = class_scores.gather(1, inverse[:, None]).squeeze(1)
        neighbor_indices = semantic_neighbors[inverse]
        hard_scores = class_scores.gather(1, neighbor_indices).max(dim=1).values
        return F.relu(
            hard_scores - positive_scores + self.semantic_batch_hard_margin
        ).mean()

    def _semantic_neighbor_rank_loss(self):
        """Preserve raw-semantic neighbourhood order in AV-to-text scores.

        Class centres avoid overweighting duplicate samples.  For every Seen
        training class, its nearest raw-word-vector classes should score above
        equally many far classes, while its own prototype remains above the
        nearest alternatives.  No validation or Unseen class is used.
        """
        labels = self.batch_labels
        if labels is None:
            raise RuntimeError(
                "semantic neighbour-rank loss requires batch class labels")
        labels = labels.reshape(-1)
        if labels.shape[0] != self.theta_a.shape[0]:
            raise ValueError(
                "semantic neighbour-rank labels must match batch size")

        class_ids, inverse = torch.unique(
            labels, sorted=True, return_inverse=True)
        class_count = int(class_ids.numel())
        if class_count < 2:
            return self.theta_a.new_zeros(())

        def class_average(embeddings):
            centres = embeddings.new_zeros(class_count, embeddings.shape[1])
            centres.index_add_(0, inverse, embeddings)
            counts = torch.bincount(inverse, minlength=class_count).to(
                dtype=embeddings.dtype, device=embeddings.device)
            return centres / counts[:, None]

        raw_text = F.normalize(class_average(self.w.detach()), dim=1)
        text_centres = F.normalize(class_average(self.theta_w), dim=1)
        audio_centres = class_average(self.theta_a)
        video_centres = class_average(self.theta_v)
        av_centres = F.normalize(0.5 * (audio_centres + video_centres), dim=1)
        learned_scores = av_centres @ text_centres.transpose(0, 1)

        raw_similarity = raw_text @ raw_text.transpose(0, 1)
        diagonal = torch.eye(
            class_count, dtype=torch.bool, device=raw_similarity.device)
        ordered = raw_similarity.masked_fill(diagonal, -torch.inf).argsort(
            dim=1, descending=True)
        # The diagonal sorts last after masking.  Remove it before taking the
        # tail; otherwise the matching class would be treated as a far class
        # and make the two ranking margins contradictory.
        ordered_nonself = ordered[:, :class_count - 1]
        neighbor_count = min(
            self.semantic_neighbor_rank_neighbors,
            max(1, (class_count - 1) // 2))
        near_indices = ordered_nonself[:, :neighbor_count]
        positive_scores = learned_scores.diagonal()[:, None]
        near_scores = learned_scores.gather(1, near_indices)
        positive_loss = F.relu(
            near_scores - positive_scores + self.semantic_neighbor_rank_margin
        ).mean()

        if class_count == 2:
            return positive_loss
        far_indices = ordered_nonself[:, -neighbor_count:]
        far_scores = learned_scores.gather(1, far_indices)
        ordering_loss = F.relu(
            far_scores[:, None, :] - near_scores[:, :, None] +
            self.semantic_neighbor_rank_margin
        ).mean()
        return 0.5 * (positive_loss + ordering_loss)

    def _semantic_mixup_loss(self):
        """Align mixed AV embeddings with virtual text prototypes.

        The model only sees examples from Seen classes.  Mixing two different
        Seen-class word vectors creates a local semantic prototype that is not
        tied to either training label.  The corresponding mixed audio/video
        embeddings are aligned to it, encouraging a smoother class manifold
        without accessing an Unseen class name or sample.
        """
        labels = self.batch_labels
        if labels is None:
            raise RuntimeError("semantic mixup loss requires batch class labels")
        labels = labels.reshape(-1)
        if labels.shape[0] != self.theta_a.shape[0]:
            raise ValueError("semantic mixup labels must match the batch size")

        different_class = labels[:, None].ne(labels[None, :])
        valid_rows = different_class.any(dim=1)
        if not valid_rows.any():
            return self.theta_a.new_zeros(())

        # Draw one partner from a different class for every valid sample.
        # Argmax over random scores avoids a Python loop and guarantees that a
        # same-class partner cannot be selected.
        random_scores = torch.rand(
            different_class.shape, device=self.theta_a.device,
            dtype=self.theta_a.dtype)
        partners = random_scores.masked_fill(~different_class, -1.0).argmax(dim=1)

        source_indices = valid_rows.nonzero(as_tuple=False).squeeze(1)
        partner_indices = partners[source_indices]
        alpha = self.semantic_mixup_alpha
        mixing = torch.distributions.Beta(alpha, alpha).sample(
            (source_indices.shape[0], 1)).to(
                device=self.theta_a.device, dtype=self.theta_a.dtype)

        def interpolate(values):
            return (mixing * values[source_indices] +
                    (1.0 - mixing) * values[partner_indices])

        # Project the mixed *raw* text embedding so this is not equivalent to
        # the existing per-sample projection loss through W_proj's nonlinearity.
        text_mix = self.W_proj(interpolate(self.w))
        audio_mix = interpolate(self.theta_a)
        video_mix = interpolate(self.theta_v)
        joint_mix = 0.5 * (audio_mix + video_mix)

        def cosine_alignment(visual):
            return 1.0 - F.cosine_similarity(visual, text_mix, dim=1).mean()

        return (cosine_alignment(audio_mix) + cosine_alignment(video_mix) +
                cosine_alignment(joint_mix)) / 3.0

    def _feature_mixup_loss(self):
        """Keep the full SNN-STFT encoder smooth between two class features.

        Unlike semantic mixup, this interpolates the paired positive and
        different-class negative before the temporal SNN, spatial TRL and
        Tucker fusion.  The interpolated text uses the same coefficient,
        creating a local pseudo-unseen semantic point without test data.
        """
        if self._feature_mixup_inputs is None:
            raise RuntimeError("feature mixup inputs are missing from the forward pass")
        if self.ahse_standardize:
            raise ValueError("feature mixup is not compatible with ahse_standardize")
        (audio, video, text, negative_audio, negative_video,
         negative_text) = self._feature_mixup_inputs
        batch_size = audio.shape[0]
        mixing = torch.distributions.Beta(
            self.feature_mixup_alpha, self.feature_mixup_alpha).sample(
                (batch_size, 1)).to(device=audio.device, dtype=audio.dtype)
        # Endpoint-biased mixing avoids treating the midpoint of two unrelated
        # clips as an equally reliable feature sample.
        mixing = torch.maximum(mixing, 1.0 - mixing)

        def interpolate(positive, negative):
            return mixing * positive + (1.0 - mixing) * negative

        mixed_audio = interpolate(audio, negative_audio)
        mixed_video = interpolate(video, negative_video)
        mixed_text = interpolate(text, negative_text)
        phi_a, snn_a = self._encode_temporal_audio(mixed_audio)
        phi_v, snn_v = self._encode_temporal_video(mixed_video)
        spatial_a, spatial_v = self._encode_spatial(mixed_audio, mixed_video)
        theta_a, theta_v = self._fuse_and_project(
            phi_a, phi_v, snn_a, snn_v, spatial_a, spatial_v)
        theta_w = self.W_proj(mixed_text)

        def cosine_alignment(embedding):
            return 1.0 - F.cosine_similarity(embedding, theta_w, dim=1).mean()

        return 0.5 * (cosine_alignment(theta_a) + cosine_alignment(theta_v))

    def _feature_debias_loss(self):
        """Regularize the optional visual semantic/residual decomposition."""
        if self._feature_debias_state is None:
            raise RuntimeError("feature debias state is missing from the positive branch")
        labels = self.batch_labels.reshape(-1)
        state = self._feature_debias_state
        semantic = state["semantic"]
        residual = state["residual"]

        reconstruction = self.criterion_reg(
            state["reconstruction"], state["source"])

        semantic_normalized = F.normalize(semantic, dim=1)
        semantic_similarity = semantic_normalized @ semantic_normalized.transpose(0, 1)
        same_class = labels[:, None].eq(labels[None, :])
        same_class.fill_diagonal_(False)
        if same_class.any():
            compactness = (1.0 - semantic_similarity[same_class]).mean()
        else:
            compactness = semantic.new_zeros(())

        # An orthogonal residual is less likely to duplicate the semantic path.
        orthogonality = (
            (F.normalize(semantic, dim=1) * F.normalize(residual, dim=1))
            .sum(dim=1).square().mean())

        positive_mask = labels[:, None].eq(labels[None, :])
        residual_logits = (
            F.normalize(state["residual_text"], dim=1) @
            F.normalize(self.theta_w.detach(), dim=1).transpose(0, 1))
        residual_logits = residual_logits / self.feature_debias_temperature
        residual_text = (
            torch.logsumexp(residual_logits, dim=1) -
            torch.logsumexp(
                residual_logits.masked_fill(~positive_mask, -torch.inf),
                dim=1)).mean()

        # Reconstruction and same-class consistency carry the main signal;
        # the two auxiliary terms are deliberately downweighted for stability.
        total = reconstruction + compactness + 0.1 * (
            orthogonality + residual_text)
        return total, {
            "feature_debias_reconstruction": reconstruction,
            "feature_debias_compactness": compactness,
            "feature_debias_orthogonality": orthogonality,
            "feature_debias_residual_text": residual_text,
        }

    def _snn_activity_floor_loss(self):
        """Penalize only final-layer firing rates below a sparse target.

        The hard-spike forward path and DTH updates remain untouched. This term
        only prevents the temporal branch from becoming silent enough to be
        bypassed by the semantic path.
        """
        if self._positive_snn_rates is None:
            raise RuntimeError("SNN activity-floor state is missing from the positive branch")
        audio_rate, video_rate = self._positive_snn_rates
        target = self.snn_min_spike_rate

        def floor_penalty(rate):
            return (F.relu(target - rate) / target).square()

        return 0.5 * (floor_penalty(audio_rate) + floor_penalty(video_rate))

    @staticmethod
    def _project_with_frozen_batchnorm(projector, features):
        """Project an auxiliary view without updating the shared BN state."""
        was_training = projector.training
        if was_training:
            projector.eval()
        try:
            return projector(features)
        finally:
            if was_training:
                projector.train()

    def _build_snn_temporal_view(self, theta_a, theta_v, theta_w):
        """Create the training-only view with semantic and spatial paths absent.

        The view begins directly at each SNN output and reuses the normal
        modality projection head.  It therefore cannot obtain information
        from the semantic encoder, LKC, Tucker core, or TRL branch.  BatchNorm
        statistics remain those of the normal fused path above.
        """
        temporal_a = self._project_with_frozen_batchnorm(
            self.A_proj, self.phi_a1)
        temporal_v = self._project_with_frozen_batchnorm(
            self.V_proj, self.phi_v1)
        self._snn_temporal_view = (temporal_a, temporal_v)
        self._snn_temporal_teachers = (
            theta_a.detach(), theta_v.detach(), theta_w.detach())

    def _snn_temporal_consistency_loss(self):
        """Align a randomly selected pure-SNN view to text and fused teachers."""
        if (self._snn_temporal_view is None or
                self._snn_temporal_teachers is None):
            raise RuntimeError("SNN temporal-view state is missing from the positive branch")
        temporal_a, temporal_v = self._snn_temporal_view
        teacher_a, teacher_v, teacher_w = self._snn_temporal_teachers
        batch_size = temporal_a.shape[0]
        if self.snn_temporal_view_fraction >= 1.0:
            selected = torch.ones(
                batch_size, device=temporal_a.device, dtype=torch.bool)
        else:
            selected = torch.rand(
                batch_size, device=temporal_a.device) < self.snn_temporal_view_fraction
            # Keep the auxiliary objective present even in a small final batch.
            if not selected.any():
                selected[torch.randint(batch_size, (1,), device=selected.device)] = True

        def view_error(temporal, fused_teacher):
            text_alignment = 1.0 - F.cosine_similarity(
                temporal, teacher_w, dim=1)
            fused_alignment = 1.0 - F.cosine_similarity(
                temporal, fused_teacher, dim=1)
            return 0.5 * (text_alignment + fused_alignment)

        audio_error = view_error(temporal_a, teacher_a)
        video_error = view_error(temporal_v, teacher_v)
        loss = 0.5 * (audio_error[selected].mean() +
                      video_error[selected].mean())
        return loss, selected.float().mean()

    def _temporal_quality_alignment_loss(self):
        """Align semantic and SNN temporal directions on both triplet views.

        The loss is train-only and uses no class or validation information. It
        regularizes the internal quality signal used by the rejected post-hoc
        fusion, so that the signal is learned consistently in both stages.
        """
        if self._temporal_quality_features is None:
            raise RuntimeError("Temporal-quality features are missing from the positive branch")
        features = self._temporal_quality_features
        pairs = ((features[0], features[1]), (features[2], features[3]),
                 (features[4], features[5]), (features[6], features[7]))
        agreement = torch.stack([
            1.0 - F.cosine_similarity(semantic, temporal, dim=1).mean()
            for semantic, temporal in pairs])
        return agreement.mean()

    def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):
        self._feature_debias_state = None
        self._feature_mixup_inputs = None
        self._positive_snn_rates = None
        self._snn_temporal_view = None
        self._snn_temporal_teachers = None
        self._spatial_reliability_state = None
        self._snn_runtime_diagnostics = {}
        # --- temporal-semantic branch (positive) ---
        # phi_* and phi_*1 are semantic_dim encoder and SNN features.
        self.phi_a, self.phi_a1 = self._encode_temporal_audio(audio)
        self._snn_runtime_diagnostics["snn_audio_spike_rate"] = (
            self.SNNbranchaudio._last_spike_rate)
        self._snn_runtime_diagnostics["snn_audio_threshold"] = (
            self.SNNbranchaudio._last_dth_threshold)
        self.phi_v, self.phi_v1 = self._encode_temporal_video(image)
        self._snn_runtime_diagnostics["snn_video_spike_rate"] = (
            self.SNNbranchvideo._last_spike_rate)
        self._snn_runtime_diagnostics["snn_video_threshold"] = (
            self.SNNbranchvideo._last_dth_threshold)
        self._positive_snn_rates = (
            self.SNNbranchaudio._last_spike_rate_for_loss,
            self.SNNbranchvideo._last_spike_rate_for_loss)
        # --- temporal-semantic branch (negative) ---
        self.phi_a_neg, self.phi_a_neg1 = self._encode_temporal_audio(negative_audio)
        self.phi_v_neg, self.phi_v_neg1 = self._encode_temporal_video(negative_image)
        if self.temporal_quality_alignment_weight > 0.0:
            self._temporal_quality_features = (
                self.phi_a, self.phi_a1, self.phi_v, self.phi_v1,
                self.phi_a_neg, self.phi_a_neg1,
                self.phi_v_neg, self.phi_v_neg1)
        self.phi_at, self.phi_vt = self._encode_spatial(audio, image)
        self.phi_at_neg, self.phi_vt_neg = self._encode_spatial(
            negative_audio, negative_image)
        # --- text / semantic projection ---
        self.w = word_embedding
        self.w_neg = negative_word_embedding
        if self.feature_mixup_weight > 0.0:
            self._feature_mixup_inputs = (
                audio, image, word_embedding, negative_audio,
                negative_image, negative_word_embedding)
        theta_w = self.W_proj(word_embedding)
        theta_w_neg = self.W_proj(negative_word_embedding)
        self.theta_w_raw = theta_w
        self.theta_w_neg_raw = theta_w_neg

        # --- STFT second-stage fusion + projection ---
        # Fuses semantic (phi) + temporal (phi_*1) per modality via LKC + Tucker.
        theta_a, theta_v = self._fuse_and_project(
            self.phi_a, self.phi_v, self.phi_a1, self.phi_v1,
            self.phi_at, self.phi_vt, collect_feature_debias=True)
        theta_a_neg, theta_v_neg = self._fuse_and_project(
            self.phi_a_neg, self.phi_v_neg, self.phi_a_neg1, self.phi_v_neg1,
            self.phi_at_neg, self.phi_vt_neg)
        if self.snn_temporal_consistency_weight > 0.0 and self.training:
            self._build_snn_temporal_view(theta_a, theta_v, theta_w)

        (self.theta_a, self.theta_v, self.theta_a_neg,
         self.theta_v_neg, self.theta_w, self.theta_w_neg) = (
            self._standardize_training_embeddings(
                theta_a, theta_v, theta_a_neg, theta_v_neg,
                theta_w, theta_w_neg))
        self.rho_w = self.D(self.theta_w)

        # --- reconstruction heads ---
        self.phi_v_rec = self.V_rec(self.theta_v)
        self.phi_a_rec = self.A_rec(self.theta_a)

        self.rho_a = self.D(self.theta_a)
        self.rho_v = self.D(self.theta_v)

    def backward(self, optimize, teacher_embeddings=None, teacher_mask=None,
                 teacher_weight=0.0):
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

        avla_contrastive = None
        if self.avla_contrastive_only:
            avla_contrastive = self._avla_contrastive_loss()
            loss_gen = avla_contrastive
        else:
            # Paper setting: L_all = 0.5 L_t + 0.5 (L_p + L_r).
            loss_gen = 0.5 * l_triplet + 0.5 * (l_projection + l_reconstruction)
        semantic_geometry = None
        if self.semantic_geometry_weight > 0.0:
            semantic_geometry = self._semantic_geometry_loss()
            loss_gen = loss_gen + self.semantic_geometry_weight * semantic_geometry
        semantic_contrastive = None
        if self.semantic_contrastive_weight > 0.0:
            semantic_contrastive = self._semantic_contrastive_loss()
            loss_gen = loss_gen + (
                self.semantic_contrastive_weight * semantic_contrastive)
        pseudo_unseen = None
        # This objective intentionally uses only the current training-stage
        # class dictionary. Validation contains held-out class ids, so it must
        # never be evaluated there or it would either fail or leak protocol
        # information into the validation loss.
        if self.pseudo_unseen_weight > 0.0 and optimize:
            pseudo_unseen = self._pseudo_unseen_episode_loss()
            loss_gen = loss_gen + self.pseudo_unseen_weight * pseudo_unseen
        snn_temporal_consistency = None
        snn_temporal_view_coverage = None
        if self.snn_temporal_consistency_weight > 0.0 and optimize:
            (snn_temporal_consistency,
             snn_temporal_view_coverage) = self._snn_temporal_consistency_loss()
            loss_gen = loss_gen + (
                self.snn_temporal_consistency_weight *
                snn_temporal_consistency)
        temporal_quality_alignment = None
        if self.temporal_quality_alignment_weight > 0.0 and optimize:
            temporal_quality_alignment = self._temporal_quality_alignment_loss()
            loss_gen = loss_gen + (
                self.temporal_quality_alignment_weight *
                temporal_quality_alignment)
        cross_modal_contrastive = None
        if self.cross_modal_contrastive_weight > 0.0:
            cross_modal_contrastive = self._cross_modal_contrastive_loss()
            loss_gen = loss_gen + (
                self.cross_modal_contrastive_weight *
                cross_modal_contrastive)
        global_prototype_contrastive = None
        if self.global_prototype_contrastive_weight > 0.0:
            global_prototype_contrastive = self._global_prototype_contrastive_loss()
            loss_gen = loss_gen + (
                self.global_prototype_contrastive_weight *
                global_prototype_contrastive)
        semantic_hard_negative = None
        if self.semantic_hard_negative_weight > 0.0:
            semantic_hard_negative = self._semantic_hard_negative_loss()
            loss_gen = loss_gen + (
                self.semantic_hard_negative_weight * semantic_hard_negative)
        semantic_batch_hard = None
        if self.semantic_batch_hard_weight > 0.0:
            semantic_batch_hard = self._semantic_batch_hard_loss()
            loss_gen = loss_gen + (
                self.semantic_batch_hard_weight * semantic_batch_hard)
        semantic_neighbor_rank = None
        if self.semantic_neighbor_rank_weight > 0.0 and optimize:
            semantic_neighbor_rank = self._semantic_neighbor_rank_loss()
            loss_gen = loss_gen + (
                self.semantic_neighbor_rank_weight * semantic_neighbor_rank)
        semantic_mixup = None
        if self.semantic_mixup_weight > 0.0:
            semantic_mixup = self._semantic_mixup_loss()
            loss_gen = loss_gen + self.semantic_mixup_weight * semantic_mixup
        feature_mixup = None
        if self.feature_mixup_weight > 0.0:
            feature_mixup = self._feature_mixup_loss()
            loss_gen = loss_gen + self.feature_mixup_weight * feature_mixup
        feature_debias = None
        feature_debias_details = None
        if self.feature_debiaser is not None:
            feature_debias, feature_debias_details = self._feature_debias_loss()
            loss_gen = loss_gen + self.feature_debias_weight * feature_debias

        snn_activity_floor = None
        if self.snn_activity_floor_weight > 0.0:
            snn_activity_floor = self._snn_activity_floor_loss()
            loss_gen = loss_gen + (
                self.snn_activity_floor_weight * snn_activity_floor)

        seen_distill = None
        seen_distill_coverage = None
        if teacher_embeddings is not None:
            if len(teacher_embeddings) != 2:
                raise ValueError(
                    "Stage B teacher embeddings must contain audio and video tensors")
            teacher_a, teacher_v = teacher_embeddings
            if teacher_mask is None:
                raise ValueError(
                    "Stage B teacher distillation requires a seen-class mask")
            teacher_mask = teacher_mask.reshape(-1).bool()
            if teacher_mask.shape[0] != self.theta_a.shape[0]:
                raise ValueError(
                    "Stage B teacher mask must match the current batch size")
            if teacher_a.shape != self.theta_a.shape or teacher_v.shape != self.theta_v.shape:
                raise ValueError(
                    "Stage B teacher embeddings must match the student modality shapes")
            if teacher_weight < 0.0:
                raise ValueError("Stage B teacher weight must be non-negative")
            seen_distill_coverage = teacher_mask.float().mean()
            if teacher_mask.any():
                audio_alignment = 1.0 - F.cosine_similarity(
                    self.theta_a[teacher_mask], teacher_a.detach()[teacher_mask],
                    dim=1).mean()
                video_alignment = 1.0 - F.cosine_similarity(
                    self.theta_v[teacher_mask], teacher_v.detach()[teacher_mask],
                    dim=1).mean()
                seen_distill = 0.5 * (audio_alignment + video_alignment)
            else:
                seen_distill = self.theta_a.new_zeros(())
            loss_gen = loss_gen + float(teacher_weight) * seen_distill

        if optimize == True:
            self.optimizer_gen.zero_grad()
            loss_gen.backward()
            self.optimizer_gen.step()

        loss = {'triplet': l_triplet, 'projection': l_projection,
                'reconstruction': l_reconstruction, 'gen': loss_gen}
        if semantic_geometry is not None:
            loss['semantic_geometry'] = semantic_geometry
        if semantic_contrastive is not None:
            loss['semantic_contrastive'] = semantic_contrastive
        if pseudo_unseen is not None:
            loss['pseudo_unseen'] = pseudo_unseen
        if snn_temporal_consistency is not None:
            loss['snn_temporal_consistency'] = snn_temporal_consistency
            loss['snn_temporal_view_coverage'] = snn_temporal_view_coverage
        if temporal_quality_alignment is not None:
            loss['temporal_quality_alignment'] = temporal_quality_alignment
        if cross_modal_contrastive is not None:
            loss['cross_modal_contrastive'] = cross_modal_contrastive
        if avla_contrastive is not None:
            loss['avla_contrastive'] = avla_contrastive
        if global_prototype_contrastive is not None:
            loss['global_prototype_contrastive'] = global_prototype_contrastive
        if semantic_hard_negative is not None:
            loss['semantic_hard_negative'] = semantic_hard_negative
        if semantic_batch_hard is not None:
            loss['semantic_batch_hard'] = semantic_batch_hard
        if semantic_neighbor_rank is not None:
            loss['semantic_neighbor_rank'] = semantic_neighbor_rank
        if semantic_mixup is not None:
            loss['semantic_mixup'] = semantic_mixup
        if feature_mixup is not None:
            loss['feature_mixup'] = feature_mixup
        if feature_debias is not None:
            loss['feature_debias'] = feature_debias
            loss.update(feature_debias_details)
        if snn_activity_floor is not None:
            loss['snn_activity_floor'] = snn_activity_floor
        if seen_distill is not None:
            loss['seen_distill'] = seen_distill
            loss['seen_distill_coverage'] = seen_distill_coverage

        loss_numeric = loss_gen

        return loss_numeric, loss

    def optimize_params(self, audio, video, cls_numeric, cls_embedding,
                        audio_negative, video_negative, negative_cls_embedding,
                        optimize=False, teacher_embeddings=None, teacher_mask=None,
                        teacher_weight=0.0):
        self.batch_labels = cls_numeric.detach()
        self.forward(audio, video, audio_negative, video_negative, cls_embedding, negative_cls_embedding)

        loss_numeric, loss = self.backward(
            optimize, teacher_embeddings=teacher_embeddings,
            teacher_mask=teacher_mask, teacher_weight=teacher_weight)

        return loss_numeric, loss

    def get_runtime_diagnostics(self):
        """Return detached scalar SNN health signals from the positive branch."""
        diagnostics = dict(self._snn_runtime_diagnostics)
        if self._spatial_reliability_state is not None:
            audio_gate, video_gate = self._spatial_reliability_state
            diagnostics.update(
                spatial_audio_gate_mean=audio_gate.mean(),
                spatial_audio_gate_std=audio_gate.std(unbiased=False),
                spatial_video_gate_mean=video_gate.mean(),
                spatial_video_gate_std=video_gate.std(unbiased=False),
            )
        return diagnostics

    def recalibrate_text_batchnorm(self, text_prototypes, mix=1.0):
        """Re-estimate ``W_proj`` BatchNorm statistics from class semantics.

        The text projection is trained on Seen-class word vectors but projects
        a Seen+Unseen class dictionary at GZSL evaluation.  Class semantics are
        available in the standard zero-shot protocol, so this evaluation-only
        diagnostic uses no audio/video example and updates no learned weight.
        """
        if self.text_projection_norm != "batchnorm":
            raise ValueError(
                "text BatchNorm recalibration requires "
                "text_projection_norm='batchnorm'")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("text BatchNorm recalibration mix must be in [0, 1]")

        text_prototypes = torch.as_tensor(
            text_prototypes, dtype=torch.float32,
            device=next(self.W_proj.parameters()).device)
        if (text_prototypes.ndim != 2 or
                text_prototypes.shape[1] != self.text_embedding_size):
            raise ValueError(
                "text prototypes for BatchNorm recalibration must have shape "
                f"(classes, {self.text_embedding_size})")
        if text_prototypes.shape[0] < 2:
            raise ValueError(
                "text BatchNorm recalibration requires at least two classes")

        batch_norms = [module for module in self.W_proj.modules()
                       if isinstance(module, nn.BatchNorm1d)]
        if not batch_norms:
            raise RuntimeError("W_proj does not contain BatchNorm layers")

        prior_statistics = [
            (module.running_mean.detach().clone(),
             module.running_var.detach().clone(),
             module.num_batches_tracked.detach().clone(), module.momentum)
            for module in batch_norms]
        dropout_states = [
            (module, module.training)
            for module in self.W_proj.modules()
            if isinstance(module, nn.Dropout)]
        was_training = self.W_proj.training

        try:
            # ``momentum=1`` records the complete task dictionary exactly.
            # Disabling dropout keeps any downstream BatchNorm deterministic.
            self.W_proj.train()
            for dropout, _ in dropout_states:
                dropout.eval()
            for batch_norm in batch_norms:
                batch_norm.momentum = 1.0
            with torch.no_grad():
                self.W_proj(text_prototypes)

            for batch_norm, (old_mean, old_var, _, _) in zip(
                    batch_norms, prior_statistics):
                batch_norm.running_mean.copy_(
                    old_mean.lerp(batch_norm.running_mean, mix))
                batch_norm.running_var.copy_(
                    old_var.lerp(batch_norm.running_var, mix))
        finally:
            for batch_norm, (_, _, old_count, old_momentum) in zip(
                    batch_norms, prior_statistics):
                batch_norm.num_batches_tracked.copy_(old_count)
                batch_norm.momentum = old_momentum
            for dropout, was_dropout_training in dropout_states:
                dropout.train(was_dropout_training)
            self.W_proj.train(was_training)

        return len(batch_norms)

    def get_embeddings(self, audio, video, embedding):
        # Inference path: only the positive branch is needed.
        # audio: (B, D_a), video: (B, D_v), embedding: (B, 300).

        phi_a, phi_a1 = self._encode_temporal_audio(audio)
        phi_v, phi_v1 = self._encode_temporal_video(video)
        phi_at, phi_vt = self._encode_spatial(audio, video)
        theta_w = self.W_proj(embedding)

        theta_a, theta_v = self._fuse_and_project(
            phi_a, phi_v, phi_a1, phi_v1, phi_at, phi_vt)
        return self._standardize_inference_embeddings(
            theta_a, theta_v, theta_w)
