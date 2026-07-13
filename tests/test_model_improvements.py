import unittest

import torch

from src.model_improvements import LIFNeuron, MSTR, TRL
from src.model_residual import ResidualMSTR, VectorTRL
from src.utils import load_model_parameters


def _small_params():
    return {
        "dim_out": 64,
        "lr": 1e-4,
        "encoder_hidden_size": 32,
        "decoder_hidden_size": 32,
        "dropout_encoder": 0.0,
        "dropout_decoder": 0.0,
        "additional_dropout": 0.0,
        "depth_transformer": 1,
        "additional_triplets_loss": True,
        "reg_loss": True,
        "momentum": 0.1,
        "first_additional_triplet": 1.0,
        "second_additional_triplet": 1.0,
        "snn_T": 4,
        "snn_tau": 2.0,
        "lkc_n_slots": 2,
        "lkc_n_heads": 8,
        "tucker_rank": 8,
        "trl_rank": 8,
        "stft_dim": 32,
    }


def _inputs(batch_size=4):
    return (
        torch.randn(batch_size, 16),
        torch.randn(batch_size, 16),
        torch.arange(batch_size),
        torch.randn(batch_size, 300),
        torch.randn(batch_size, 16),
        torch.randn(batch_size, 16),
        torch.randn(batch_size, 300),
    )


def test_lif_has_hard_forward_and_sigmoid_surrogate_gradient():
    current = torch.tensor([-1.0, 0.5, 1.0, 1.5], requires_grad=True)
    neuron = LIFNeuron(tau=1.0, v_threshold=1.0, surrogate_alpha=4.0)

    spike = neuron(current)
    torch.testing.assert_close(spike, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    spike.sum().backward()

    surrogate = torch.sigmoid(4.0 * (current.detach() - 1.0))
    expected_gradient = 4.0 * surrogate * (1.0 - surrogate)
    torch.testing.assert_close(current.grad, expected_gradient)
    neuron.reset()


def test_full_model_backward_reaches_every_parameter():
    torch.manual_seed(0)
    model = MSTR(_small_params(), input_size_audio=16, input_size_video=16)
    loss, details = model.optimize_params(*_inputs(), optimize=False)

    assert torch.isfinite(loss)
    assert set(details) == {"triplet", "projection", "reconstruction", "gen"}
    loss.backward()

    missing = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    non_finite = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    assert missing == []
    assert non_finite == []
    assert model.lkc.alpha_logit.grad.abs().item() > 0
    assert model.trl_a.core.grad is not None
    assert model.trl_a.core.grad.abs().sum().item() > 0
    assert model.trl_v.core.grad is not None
    assert model.trl_v.core.grad.abs().sum().item() > 0
    assert model.tucker_fusion.G_a.shape == (8, 8, 8)


def test_trl_matches_dense_tucker_regression_for_dynamic_batches():
    torch.manual_seed(2)
    trl = TRL(
        input_size=(1, 5, 2, 1), ranks=(3, 2, 1, 4),
        output_size=(1, 6))
    x = torch.randn(7, 5, 2, 1)

    dense_weight = torch.einsum(
        "rstu,ir,js,kt,ou->ijko",
        trl.core, trl.factors[0], trl.factors[1],
        trl.factors[2], trl.factors[3])
    expected = torch.einsum("bijk,ijko->bo", x, dense_weight) + trl.bias

    torch.testing.assert_close(trl(x), expected, rtol=1e-5, atol=1e-6)
    assert trl(x[:3]).shape == (3, 6)


def test_300_dim_control_configuration_uses_six_lkc_heads():
    params = _small_params()
    params["stft_dim"] = 300
    params["lkc_n_heads"] = 6
    model = MSTR(params, input_size_audio=16, input_size_video=16)

    assert model.lkc.sa.num_heads == 6


def test_inference_is_deterministic_and_resets_snn_state():
    torch.manual_seed(1)
    model = MSTR(_small_params(), input_size_audio=16, input_size_video=16).eval()
    audio, video, _, text, *_ = _inputs()

    with torch.no_grad():
        first = model.get_embeddings(audio, video, text)
        second = model.get_embeddings(audio, video, text)

    for first_embedding, second_embedding in zip(first, second):
        torch.testing.assert_close(first_embedding, second_embedding, rtol=0, atol=0)
    assert model.SNNbranchaudio.lif1.v is None
    assert model.SNNbranchvideo.lif1.v is None
    assert model.SNNbranchaudio.lif1.v_threshold.item() == 1.0


def test_tsf_weights_depend_on_each_samples_spike_history():
    model = MSTR(_small_params(), input_size_audio=16, input_size_video=16).eval()
    outs = torch.zeros(2, 4, 3)
    outs[0, 0, 0] = 1.0
    outs[1, 3, 0] = 1.0

    _, weights = model._time_step_fusion(outs)

    assert weights.shape == (2, 4)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
    assert not torch.equal(weights[0], weights[1])
    assert weights[0].argmax().item() == 0
    assert weights[1].argmax().item() == 3


def test_checkpoint_loading_rejects_partial_state():
    model = MSTR(_small_params(), input_size_audio=16, input_size_video=16)
    state = model.state_dict()
    incomplete_state = {name: value for name, value in state.items()
                        if name != "tucker_fusion.G_a"}

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "missing=.*G_a"):
        load_model_parameters(model, incomplete_state, strict=True)


def test_vector_trl_has_stable_nonzero_gradients_after_gate_opens():
    torch.manual_seed(3)
    trl = VectorTRL(input_size=16, output_size=12, rank=6)
    gate = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam(list(trl.parameters()) + [gate], lr=1e-3)
    x = torch.randn(8, 16)

    for _ in range(2):
        optimizer.zero_grad()
        output = 0.25 * torch.tanh(gate) * torch.nn.functional.layer_norm(
            trl(x), (12,))
        loss = (output - torch.randn_like(output)).pow(2).mean()
        loss.backward()
        optimizer.step()

    assert gate.abs().item() > 0
    assert trl.input_factor.weight.grad.abs().sum().item() > 0
    assert trl.output_factor.weight.grad.abs().sum().item() > 0
    assert all(torch.isfinite(parameter).all() for parameter in trl.parameters())


def test_frozen_residual_backbone_keeps_batchnorm_statistics_fixed():
    params = _small_params()
    params.update(fusion_mode="residual", vector_trl_rank=8,
                  trl_gate_scale=0.25, backbone_lr_scale=0.0)
    model = ResidualMSTR(params, input_size_audio=16, input_size_video=16)
    model.train()
    running_mean = model.A_enc.fc[1].running_mean.clone()

    model.optimize_params(*_inputs(), optimize=True)

    torch.testing.assert_close(model.A_enc.fc[1].running_mean, running_mean)
    assert not model.A_enc.training
    trainable = {name for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    assert trainable == {
        "trl_gate_a", "trl_gate_v",
        "vector_trl_a.bias", "vector_trl_a.input_factor.weight",
        "vector_trl_a.output_factor.weight", "vector_trl_v.bias",
        "vector_trl_v.input_factor.weight",
        "vector_trl_v.output_factor.weight",
    }


class ModelImprovementsTest(unittest.TestCase):
    test_lif_has_hard_forward_and_sigmoid_surrogate_gradient = staticmethod(
        test_lif_has_hard_forward_and_sigmoid_surrogate_gradient
    )
    test_full_model_backward_reaches_every_parameter = staticmethod(
        test_full_model_backward_reaches_every_parameter
    )
    test_trl_matches_dense_tucker_regression_for_dynamic_batches = staticmethod(
        test_trl_matches_dense_tucker_regression_for_dynamic_batches
    )
    test_300_dim_control_configuration_uses_six_lkc_heads = staticmethod(
        test_300_dim_control_configuration_uses_six_lkc_heads
    )
    test_inference_is_deterministic_and_resets_snn_state = staticmethod(
        test_inference_is_deterministic_and_resets_snn_state
    )
    test_tsf_weights_depend_on_each_samples_spike_history = staticmethod(
        test_tsf_weights_depend_on_each_samples_spike_history
    )
    test_checkpoint_loading_rejects_partial_state = staticmethod(
        test_checkpoint_loading_rejects_partial_state
    )
    test_vector_trl_has_stable_nonzero_gradients_after_gate_opens = staticmethod(
        test_vector_trl_has_stable_nonzero_gradients_after_gate_opens
    )
    test_frozen_residual_backbone_keeps_batchnorm_statistics_fixed = staticmethod(
        test_frozen_residual_backbone_keeps_batchnorm_statistics_fixed
    )


if __name__ == "__main__":
    unittest.main()
