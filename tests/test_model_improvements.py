import unittest

import torch
import numpy as np

from src.model_improvements import (CrossModalResidualGate, EmbeddingNet,
                                    GlobalLocalPool, LIFNeuron, MSTR, TRL,
                                    RunningFeatureStandardizer,
                                    SpatialReliabilityGate, StableVectorTRL,
                                    TemporalSemanticTuckerFusion,
                                    gradient_reverse)
from src.model_mstr_baseline import MSTRBaseline
from src.model_residual import ResidualMSTR, VectorTRL
from src.utils import load_model_parameters
from src.dataset import ContrastiveDataset


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


def test_text_projection_layernorm_has_no_batch_statistics():
    projection = EmbeddingNet(
        input_size=300, output_size=64, dropout=0.0, use_bn=True,
        momentum=0.1, normalization="layernorm")
    projection.train()
    output = projection(torch.randn(1, 300))

    assert output.shape == (1, 64)
    assert any(isinstance(module, torch.nn.LayerNorm)
               for module in projection.modules())
    assert not any(isinstance(module, torch.nn.BatchNorm1d)
                   for module in projection.modules())


def test_text_batchnorm_recalibration_uses_only_class_semantics():
    torch.manual_seed(22)
    model = MSTR(_small_params(), input_size_audio=16, input_size_video=16).eval()
    batch_norms = [module for module in model.W_proj.modules()
                   if isinstance(module, torch.nn.BatchNorm1d)]
    old_statistics = [
        (module.running_mean.clone(), module.num_batches_tracked.clone())
        for module in batch_norms]

    count = model.recalibrate_text_batchnorm(torch.randn(5, 300))

    assert count == len(batch_norms) > 0
    assert model.W_proj.training is False
    assert all(torch.equal(module.num_batches_tracked, old_count)
               for module, (_, old_count) in zip(batch_norms, old_statistics))
    assert any(not torch.equal(module.running_mean, old_mean)
               for module, (old_mean, _) in zip(batch_norms, old_statistics))
    assert all(torch.isfinite(module.running_var).all() for module in batch_norms)

    layernorm_params = _small_params()
    layernorm_params["text_projection_norm"] = "layernorm"
    layernorm_model = MSTR(layernorm_params, input_size_audio=16,
                           input_size_video=16)
    try:
        layernorm_model.recalibrate_text_batchnorm(torch.randn(5, 300))
    except ValueError as error:
        assert "batchnorm" in str(error)
    else:
        raise AssertionError("LayerNorm text projection must reject BN calibration")


def test_gradient_reverse_flips_only_the_backward_signal():
    features = torch.tensor([1.0, -2.0], requires_grad=True)
    reversed_features = gradient_reverse(features, scale=0.25)
    torch.testing.assert_close(reversed_features, features)
    reversed_features.sum().backward()
    torch.testing.assert_close(features.grad, torch.tensor([-0.25, -0.25]))


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
    diagnostics = model.get_runtime_diagnostics()
    assert set(diagnostics) == {
        "snn_audio_spike_rate", "snn_audio_threshold",
        "snn_video_spike_rate", "snn_video_threshold",
    }
    assert all(torch.isfinite(value).all() for value in diagnostics.values())
    assert all(not value.requires_grad for value in diagnostics.values())


def test_global_prototype_contrastive_loss_uses_nonpersistent_text_dictionary():
    torch.manual_seed(19)
    params = _small_params()
    params.update(
        global_prototype_contrastive_weight=0.01,
        global_prototype_contrastive_temperature=0.1)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    model.set_global_text_prototypes(
        torch.randn(5, 300), torch.tensor([0, 1, 2, 3, 4]))

    loss, details = model.optimize_params(*_inputs(), optimize=False)

    assert torch.isfinite(loss)
    assert "global_prototype_contrastive" in details
    assert torch.isfinite(details["global_prototype_contrastive"])
    assert "_global_text_prototypes" not in model.state_dict()
    assert "_global_prototype_class_ids" not in model.state_dict()
    loss.backward()
    assert model.W_proj.fc[0].weight.grad is not None
    assert model.A_proj.fc[0].weight.grad is not None
    assert model.V_proj.fc[0].weight.grad is not None


def test_cross_modal_contrastive_loss_is_finite_and_reaches_snn_encoders():
    torch.manual_seed(41)
    params = _small_params()
    params.update(
        cross_modal_contrastive_weight=0.005,
        cross_modal_contrastive_temperature=0.1)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    inputs = list(_inputs(batch_size=4))
    inputs[2] = torch.tensor([0, 0, 1, 1])

    loss, details = model.optimize_params(*inputs, optimize=False)

    assert torch.isfinite(loss)
    assert "cross_modal_contrastive" in details
    assert details["cross_modal_contrastive"].item() >= 0.0
    loss.backward()
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert torch.isfinite(model.SNNbranchaudio.fc1.weight.grad).all()
    assert torch.isfinite(model.SNNbranchvideo.fc1.weight.grad).all()


def test_pseudo_unseen_episode_loss_uses_train_dictionary_and_reaches_snn():
    torch.manual_seed(47)
    params = _small_params()
    params.update(
        pseudo_unseen_weight=0.05,
        pseudo_unseen_temperature=0.15,
        pseudo_unseen_class_fraction=0.5,
        pseudo_unseen_min_classes=2)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    model.set_pseudo_unseen_text_prototypes(
        torch.randn(6, 300), torch.arange(6))
    inputs = list(_inputs(batch_size=8))
    inputs[2] = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss, details = model.optimize_params(*inputs, optimize=True)

    assert torch.isfinite(loss)
    assert "pseudo_unseen" in details
    assert details["pseudo_unseen"].item() > 0.0
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert torch.isfinite(model.SNNbranchaudio.fc1.weight.grad).all()
    assert torch.isfinite(model.SNNbranchvideo.fc1.weight.grad).all()


def test_snn_temporal_consistency_is_train_only_and_reaches_snn():
    torch.manual_seed(53)
    params = _small_params()
    params.update(
        snn_temporal_consistency_weight=0.01,
        snn_temporal_view_fraction=1.0)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    batch_norms = [module for module in model.A_proj.modules()
                   if isinstance(module, torch.nn.BatchNorm1d)]
    before = [module.num_batches_tracked.clone() for module in batch_norms]

    loss, details = model.optimize_params(*_inputs(), optimize=True)

    assert torch.isfinite(loss)
    assert "snn_temporal_consistency" in details
    assert details["snn_temporal_consistency"].item() >= 0.0
    assert details["snn_temporal_view_coverage"].item() == 1.0
    assert all(module.num_batches_tracked == count + 2
               for module, count in zip(batch_norms, before))
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert torch.isfinite(model.SNNbranchaudio.fc1.weight.grad).all()
    assert torch.isfinite(model.SNNbranchvideo.fc1.weight.grad).all()

    model.eval()
    _, eval_details = model.optimize_params(*_inputs(), optimize=False)
    assert "snn_temporal_consistency" not in eval_details


def test_temporal_quality_alignment_is_train_only_and_reaches_snn():
    torch.manual_seed(59)
    params = _small_params()
    params["temporal_quality_alignment_weight"] = 0.02
    model = MSTR(params, input_size_audio=16, input_size_video=16)

    loss, details = model.optimize_params(*_inputs(), optimize=True)

    assert torch.isfinite(loss)
    assert "temporal_quality_alignment" in details
    assert details["temporal_quality_alignment"].item() >= 0.0
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert torch.isfinite(model.SNNbranchaudio.fc1.weight.grad).all()
    assert torch.isfinite(model.SNNbranchvideo.fc1.weight.grad).all()

    model.eval()
    _, eval_details = model.optimize_params(*_inputs(), optimize=False)
    assert "temporal_quality_alignment" not in eval_details


def test_semantic_batch_hard_loss_is_finite_and_reaches_snn_encoders():
    torch.manual_seed(43)
    params = _small_params()
    params.update(
        semantic_batch_hard_weight=0.02,
        semantic_batch_hard_margin=0.1,
        semantic_batch_hard_neighbors=2)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    inputs = list(_inputs(batch_size=6))
    inputs[2] = torch.tensor([0, 0, 1, 1, 2, 2])

    loss, details = model.optimize_params(*inputs, optimize=False)

    assert torch.isfinite(loss)
    assert "semantic_batch_hard" in details
    assert details["semantic_batch_hard"].item() >= 0.0
    loss.backward()
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert torch.isfinite(model.SNNbranchaudio.fc1.weight.grad).all()
    assert torch.isfinite(model.SNNbranchvideo.fc1.weight.grad).all()


def test_semantic_neighbor_rank_is_train_only_and_reaches_snn_encoders():
    torch.manual_seed(47)
    params = _small_params()
    params.update(
        semantic_neighbor_rank_weight=0.02,
        semantic_neighbor_rank_margin=0.05,
        semantic_neighbor_rank_neighbors=2)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    inputs = list(_inputs(batch_size=6))
    inputs[2] = torch.tensor([0, 0, 1, 1, 2, 2])

    loss, details = model.optimize_params(*inputs, optimize=True)

    assert torch.isfinite(loss)
    assert "semantic_neighbor_rank" in details
    assert details["semantic_neighbor_rank"].item() >= 0.0
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert torch.isfinite(model.SNNbranchaudio.fc1.weight.grad).all()
    assert torch.isfinite(model.SNNbranchvideo.fc1.weight.grad).all()

    model.eval()
    _, eval_details = model.optimize_params(*inputs, optimize=False)
    assert "semantic_neighbor_rank" not in eval_details


def test_semantic_neighbor_rank_excludes_matching_class_from_far_set():
    params = _small_params()
    params.update(
        semantic_neighbor_rank_weight=0.02,
        semantic_neighbor_rank_margin=0.05,
        semantic_neighbor_rank_neighbors=5)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    model.batch_labels = torch.tensor([0, 1, 2])

    angle = torch.deg2rad(torch.tensor(20.0))
    geometry = torch.tensor([
        [1.0, 0.0],
        [torch.cos(angle), torch.sin(angle)],
        [-1.0, 0.0],
    ])
    raw_text = torch.zeros(3, 300)
    raw_text[:, :2] = geometry
    learned = torch.zeros(3, 64)
    learned[:, :2] = geometry
    model.w = raw_text
    model.theta_w = learned
    model.theta_a = learned.clone()
    model.theta_v = learned.clone()

    loss = model._semantic_neighbor_rank_loss()

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-7)


def test_seen_teacher_distillation_is_masked_and_finite():
    torch.manual_seed(31)
    student = MSTR(_small_params(), input_size_audio=16, input_size_video=16)
    teacher = MSTR(_small_params(), input_size_audio=16, input_size_video=16).eval()
    inputs = _inputs(batch_size=4)
    with torch.no_grad():
        teacher_embeddings = teacher.get_embeddings(
            inputs[0], inputs[1], inputs[3])[:2]
    seen_mask = torch.tensor([True, False, True, False])

    loss, details = student.optimize_params(
        *inputs, optimize=False, teacher_embeddings=teacher_embeddings,
        teacher_mask=seen_mask, teacher_weight=0.02)

    assert torch.isfinite(loss)
    assert set(("seen_distill", "seen_distill_coverage")).issubset(details)
    assert torch.isfinite(details["seen_distill"])
    torch.testing.assert_close(
        details["seen_distill_coverage"], torch.tensor(0.5))
    loss.backward()
    assert student.A_proj.fc[0].weight.grad is not None
    assert student.V_proj.fc[0].weight.grad is not None


def test_snn_activity_floor_is_opt_in_and_reaches_snn_parameters():
    torch.manual_seed(32)
    params = _small_params()
    params.update(snn_activity_floor_weight=0.02, snn_min_spike_rate=0.95)
    model = MSTR(params, input_size_audio=16, input_size_video=16)

    loss, details = model.optimize_params(*_inputs(), optimize=False)

    assert torch.isfinite(loss)
    assert "snn_activity_floor" in details
    assert details["snn_activity_floor"].item() > 0.0
    loss.backward()
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert torch.isfinite(model.SNNbranchaudio.fc1.weight.grad).all()
    assert torch.isfinite(model.SNNbranchvideo.fc1.weight.grad).all()


def test_membrane_readout_preserves_hard_spike_diagnostics_and_is_opt_in():
    torch.manual_seed(37)
    params = _small_params()
    params["snn_membrane_readout_scale"] = 0.2
    model = MSTR(params, input_size_audio=16, input_size_video=16)

    loss, _ = model.optimize_params(*_inputs(), optimize=False)

    assert torch.isfinite(loss)
    diagnostics = model.get_runtime_diagnostics()
    for key in ("snn_audio_spike_rate", "snn_video_spike_rate"):
        assert 0.0 <= diagnostics[key].item() <= 1.0
    loss.backward()
    assert model.SNNbranchaudio.membrane_readout_gate.grad is not None
    assert model.SNNbranchvideo.membrane_readout_gate.grad is not None
    assert model.SNNbranchaudio.fc3.weight.grad is not None
    assert model.SNNbranchvideo.fc3.weight.grad is not None

    default_model = MSTR(_small_params(), input_size_audio=16, input_size_video=16)
    assert default_model.SNNbranchaudio.membrane_readout_gate is None
    assert default_model.SNNbranchvideo.membrane_readout_gate is None
    assert not any("membrane_readout_gate" in name
                   for name in default_model.state_dict())


def test_cross_modal_residual_gate_is_bounded_and_sample_adaptive():
    torch.manual_seed(14)
    gate = CrossModalResidualGate(32, dropout=0.0, residual_scale=0.2).eval()
    audio = torch.randn(4, 32)
    video = torch.randn(4, 32)

    audio_out, video_out = gate(audio, video)
    assert audio_out.shape == audio.shape
    assert video_out.shape == video.shape
    assert torch.isfinite(audio_out).all()
    assert torch.isfinite(video_out).all()

    altered_audio, altered_video = gate(audio, video + 0.5)
    assert not torch.allclose(audio_out, altered_audio)
    assert not torch.allclose(video_out, altered_video)


def test_cross_modal_residual_is_opt_in_and_receives_gradients():
    torch.manual_seed(15)
    params = _small_params()
    params.update(cross_modal_residual=True, cross_modal_residual_scale=0.2)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    assert model.cross_modal_residual is not None
    optimizer_ids = {
        id(parameter)
        for group in model.optimizer_gen.param_groups
        for parameter in group["params"]
    }
    assert all(id(parameter) in optimizer_ids
               for parameter in model.cross_modal_residual.parameters())

    loss, _ = model.optimize_params(*_inputs(), optimize=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.cross_modal_residual.audio_from_video[1].weight.grad is not None
    assert torch.isfinite(
        model.cross_modal_residual.audio_from_video[1].weight.grad).all()

    default_model = MSTR(_small_params(), input_size_audio=16, input_size_video=16)
    assert default_model.cross_modal_residual is None
    assert not any(name.startswith("cross_modal_residual")
                   for name in default_model.state_dict())


def test_ahse_standardizer_uses_training_statistics_at_inference():
    torch.manual_seed(11)
    standardizer = RunningFeatureStandardizer(3, momentum=0.5)
    first = torch.tensor([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]])
    second = torch.tensor([[5.0, 10.0, 12.0], [7.0, 14.0, 16.0]])
    standardized = standardizer.forward_group(first, second)
    joined = torch.cat(standardized, dim=0)
    torch.testing.assert_close(joined.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(joined.var(dim=0, unbiased=False), torch.ones(3), atol=1e-5, rtol=0.0)
    assert standardizer.num_batches_tracked.item() == 1

    standardizer.eval()
    evaluation = standardizer.forward_group(first)[0]
    repeated = standardizer.forward_group(first)[0]
    torch.testing.assert_close(evaluation, repeated)
    assert standardizer.num_batches_tracked.item() == 1


def test_ahse_standardization_is_wired_into_mstr_without_changing_default_state():
    torch.manual_seed(12)
    params = _small_params()
    params["ahse_standardize"] = True
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    loss, _ = model.optimize_params(*_inputs(), optimize=False)
    assert torch.isfinite(loss)
    assert model.av_standardizer.num_batches_tracked.item() == 1
    assert model.text_standardizer.num_batches_tracked.item() == 1

    model.eval()
    embeddings = model.get_embeddings(
        torch.randn(4, 16), torch.randn(4, 16), torch.randn(4, 300))
    assert all(torch.isfinite(embedding).all() for embedding in embeddings)

    default_model = MSTR(_small_params(), input_size_audio=16, input_size_video=16)
    assert not any(name.startswith("av_standardizer")
                   for name in default_model.state_dict())


def test_semantic_geometry_regularizer_preserves_text_relations_with_gradients():
    torch.manual_seed(13)
    params = _small_params()
    params["semantic_geometry_weight"] = 0.1
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    loss, details = model.optimize_params(*_inputs(), optimize=False)

    assert torch.isfinite(loss)
    assert "semantic_geometry" in details
    assert torch.isfinite(details["semantic_geometry"])
    loss.backward()
    assert model.W_proj.fc[0].weight.grad is not None
    assert torch.isfinite(model.W_proj.fc[0].weight.grad).all()


def test_semantic_contrastive_loss_is_finite_and_reaches_modal_text_encoders():
    torch.manual_seed(16)
    params = _small_params()
    params.update(
        semantic_contrastive_weight=0.02,
        semantic_contrastive_temperature=0.1)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    inputs = list(_inputs())
    inputs[2] = torch.tensor([0, 0, 1, 1])
    loss, details = model.optimize_params(*inputs, optimize=False)

    assert torch.isfinite(loss)
    assert "semantic_contrastive" in details
    assert torch.isfinite(details["semantic_contrastive"])
    assert details["semantic_contrastive"].item() > 0.0
    loss.backward()
    assert model.A_proj.fc[0].weight.grad is not None
    assert model.V_proj.fc[0].weight.grad is not None
    assert model.W_proj.fc[0].weight.grad is not None


def test_avla_contrastive_replaces_mstr_objective_and_reaches_snn_encoder():
    torch.manual_seed(25)
    params = _small_params()
    params.update(avla_contrastive_only=True, avla_temperature=0.1)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    inputs = list(_inputs())
    inputs[2] = torch.tensor([0, 0, 1, 1])
    loss, details = model.optimize_params(*inputs, optimize=False)

    assert torch.isfinite(loss)
    assert "avla_contrastive" in details
    torch.testing.assert_close(loss, details["avla_contrastive"])
    loss.backward()
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert model.A_proj.fc[0].weight.grad is not None
    assert model.V_proj.fc[0].weight.grad is not None
    assert model.W_proj.fc[0].weight.grad is not None


def test_semantic_hard_negative_loss_uses_different_class_text_with_gradients():
    torch.manual_seed(18)
    params = _small_params()
    params.update(
        semantic_hard_negative_weight=0.05,
        semantic_hard_negative_margin=0.1)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    inputs = list(_inputs())
    inputs[2] = torch.tensor([0, 0, 1, 1])
    # Classes 0 and 1 are deliberately close in raw text space, so the loss
    # has a valid non-self hard negative for every sample.
    inputs[3][1] = inputs[3][0] + 0.01
    inputs[3][3] = inputs[3][2] + 0.01
    loss, details = model.optimize_params(*inputs, optimize=False)

    assert torch.isfinite(loss)
    assert "semantic_hard_negative" in details
    assert torch.isfinite(details["semantic_hard_negative"])
    loss.backward()
    assert model.A_proj.fc[0].weight.grad is not None
    assert model.V_proj.fc[0].weight.grad is not None
    assert model.W_proj.fc[0].weight.grad is not None


def test_semantic_mixup_aligns_different_class_manifolds_with_gradients():
    torch.manual_seed(33)
    params = _small_params()
    params.update(semantic_mixup_weight=0.02, semantic_mixup_alpha=1.0)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    inputs = list(_inputs())
    inputs[2] = torch.tensor([0, 0, 1, 1])
    loss, details = model.optimize_params(*inputs, optimize=False)

    assert torch.isfinite(loss)
    assert "semantic_mixup" in details
    assert torch.isfinite(details["semantic_mixup"])
    assert details["semantic_mixup"].item() >= 0.0
    loss.backward()
    assert model.A_proj.fc[0].weight.grad is not None
    assert model.V_proj.fc[0].weight.grad is not None
    assert model.W_proj.fc[0].weight.grad is not None


def test_feature_mixup_runs_through_snn_stft_and_reaches_encoders():
    torch.manual_seed(34)
    params = _small_params()
    params.update(feature_mixup_weight=0.01, feature_mixup_alpha=0.2)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    loss, details = model.optimize_params(*_inputs(), optimize=False)

    assert torch.isfinite(loss)
    assert "feature_mixup" in details
    assert torch.isfinite(details["feature_mixup"])
    loss.backward()
    assert model.SNNbranchaudio.fc1.weight.grad is not None
    assert model.SNNbranchvideo.fc1.weight.grad is not None
    assert model.A_proj.fc[0].weight.grad is not None
    assert model.V_proj.fc[0].weight.grad is not None
    assert model.W_proj.fc[0].weight.grad is not None


def test_feature_debias_is_opt_in_and_trains_semantic_residual_paths():
    torch.manual_seed(17)
    params = _small_params()
    params.update(feature_debias_weight=0.05, feature_debias_temperature=0.1)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    inputs = list(_inputs())
    inputs[2] = torch.tensor([0, 0, 1, 1])
    loss, details = model.optimize_params(*inputs, optimize=False)

    assert model.feature_debiaser is not None
    assert torch.isfinite(loss)
    for key in (
            "feature_debias", "feature_debias_reconstruction",
            "feature_debias_compactness", "feature_debias_orthogonality",
            "feature_debias_residual_text"):
        assert key in details
        assert torch.isfinite(details[key])

    loss.backward()
    assert model.feature_debiaser.semantic_encoder[1].weight.grad is not None
    assert model.feature_debiaser.residual_encoder[1].weight.grad is not None
    assert model.feature_debiaser.residual_text_probe[1].weight.grad is not None

    default_model = MSTR(_small_params(), input_size_audio=16, input_size_video=16)
    assert default_model.feature_debiaser is None
    assert not any(name.startswith("feature_debiaser")
                   for name in default_model.state_dict())


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


def test_snn_embeddings_are_invariant_to_batch_companions():
    """DTH must not let unrelated samples change an embedding at inference."""
    torch.manual_seed(123)
    model = MSTR(_small_params(), input_size_audio=16, input_size_video=16).eval()
    audio = torch.randn(1, 16)
    video = torch.randn(1, 16)
    text = torch.randn(1, 300)
    other_audio = torch.randn(1, 16) * 4.0 + 2.0
    other_video = torch.randn(1, 16) * 4.0 - 2.0
    other_text = torch.randn(1, 300)

    with torch.no_grad():
        alone = model.get_embeddings(audio, video, text)
        with_companion = model.get_embeddings(
            torch.cat((audio, other_audio)),
            torch.cat((video, other_video)),
            torch.cat((text, other_text)))

    for single, batched in zip(alone, with_companion):
        torch.testing.assert_close(single[0], batched[0], atol=1e-6, rtol=1e-5)


def test_legacy_batch_dth_restores_batch_companion_dependence():
    """The experimental legacy flag reproduces the historical scalar DTH."""
    torch.manual_seed(123)
    params = _small_params()
    params['legacy_batch_dth'] = True
    model = MSTR(params, input_size_audio=16, input_size_video=16).eval()
    audio = torch.randn(1, 16)
    video = torch.randn(1, 16)
    text = torch.randn(1, 300)
    other_audio = torch.randn(1, 16) * 4.0 + 2.0
    other_video = torch.randn(1, 16) * 4.0 - 2.0
    other_text = torch.randn(1, 300)

    with torch.no_grad():
        alone = model.get_embeddings(audio, video, text)
        with_companion = model.get_embeddings(
            torch.cat((audio, other_audio)),
            torch.cat((video, other_video)),
            torch.cat((text, other_text)))

    assert not torch.allclose(alone[0][0], with_companion[0][0], atol=1e-6)


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


def test_stft_vector_trl_is_opt_in_sample_dependent_and_trainable():
    baseline = MSTR(_small_params(), input_size_audio=16, input_size_video=16)
    assert isinstance(baseline.trl_a, TRL)

    params = _small_params()
    params.update(stft_vector_trl=True, vector_trl_rank=6)
    model = MSTR(params, input_size_audio=16, input_size_video=16)
    assert isinstance(model.trl_a, StableVectorTRL)
    audio = torch.randn(8, 16)
    video = torch.randn(8, 16)

    spatial_a, spatial_v = model._encode_spatial(audio, video)
    assert spatial_a.std(dim=0).mean().item() > 0.0
    assert spatial_v.std(dim=0).mean().item() > 0.0
    (spatial_a.square().mean() + spatial_v.square().mean()).backward()

    for trl in (model.trl_a, model.trl_v):
        assert trl.input_factor.weight.grad.abs().sum().item() > 0.0
        assert trl.output_factor.weight.grad.abs().sum().item() > 0.0
        assert all(torch.isfinite(parameter).all()
                   for parameter in trl.parameters())

    model.zero_grad(set_to_none=True)
    loss, _ = model.optimize_params(*_inputs(), optimize=False)
    loss.backward()
    for trl in (model.trl_a, model.trl_v):
        assert trl.input_factor.weight.grad.abs().sum().item() > 0.0
        assert trl.output_factor.weight.grad.abs().sum().item() > 0.0


def test_spatial_reliability_gate_is_bounded_and_sample_adaptive():
    gate = SpatialReliabilityGate(initial_gate=0.25)
    semantic = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    spatial = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    initial = gate(semantic, spatial)
    torch.testing.assert_close(initial, torch.full((2, 1), 0.25))
    with torch.no_grad():
        gate.projection.weight.copy_(torch.tensor([[2.0, 0.0]]))
    adaptive = gate(semantic, spatial)

    assert torch.all((adaptive > 0.0) & (adaptive < 1.0))
    assert adaptive[0].item() > adaptive[1].item()


def test_spatial_reliability_endpoints_bypass_or_restore_spatial_branch():
    torch.manual_seed(41)
    fusion = TemporalSemanticTuckerFusion(
        dim=16, rank=4, dropout=0.0).eval()
    components = [torch.randn(3, 16) for _ in range(6)]
    R_a, S_a, R_v, S_v, P_a, P_v = components
    zeros = (torch.zeros(3, 1), torch.zeros(3, 1))
    ones = (torch.ones(3, 1), torch.ones(3, 1))

    with torch.no_grad():
        full = fusion(*components)
        restored = fusion(*components, spatial_reliability=ones)
        bypassed = fusion(*components, spatial_reliability=zeros)
        changed_spatial = fusion(
            R_a, S_a, R_v, S_v, P_a * 7.0, P_v * -5.0,
            spatial_reliability=zeros)

    torch.testing.assert_close(restored[0], full[0])
    torch.testing.assert_close(restored[1], full[1])
    torch.testing.assert_close(changed_spatial[0], bypassed[0])
    torch.testing.assert_close(changed_spatial[1], bypassed[1])


def test_stft_spatial_reliability_gate_reaches_full_training_objective():
    params = _small_params()
    params.update(
        stft_vector_trl=True, vector_trl_rank=6,
        stft_spatial_reliability_gate=True)
    model = MSTR(params, input_size_audio=16, input_size_video=16)

    loss, _ = model.optimize_params(*_inputs(), optimize=False)
    loss.backward()

    gate = model.spatial_reliability_gate
    assert gate.projection.weight.grad.abs().sum().item() > 0.0
    diagnostics = model.get_runtime_diagnostics()
    assert 0.0 < diagnostics["spatial_audio_gate_mean"].item() < 1.0
    assert 0.0 < diagnostics["spatial_video_gate_mean"].item() < 1.0
    assert torch.isfinite(gate.projection.weight.grad).all()


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


def test_mstr_baseline_variants_match_released_and_paper_attention_layouts():
    paper_params = _small_params()
    paper_params.update(fusion_mode="mstr_paper", trl_rank=8, snn_T=3)
    paper = MSTRBaseline(
        paper_params, input_size_audio=16, input_size_video=16)
    released_params = dict(paper_params, fusion_mode="mstr_released")
    released = MSTRBaseline(
        released_params, input_size_audio=16, input_size_video=16)

    assert paper.cross_attention.layers[0][0].fn.heads == 8
    assert released.cross_attention.layers[0][0].fn.heads == 3
    assert paper.T == 3
    assert released.T == 10


def test_paper_mstr_baseline_supports_dynamic_batches_and_backward():
    params = _small_params()
    params.update(fusion_mode="mstr_paper", trl_rank=8, snn_T=2)
    model = MSTRBaseline(params, input_size_audio=16, input_size_video=16)

    loss, details = model.optimize_params(*_inputs(batch_size=3), optimize=False)
    assert torch.isfinite(loss)
    assert set(details) == {
        "gen", "reconstruction", "cross", "word", "additional", "reg"
    }
    loss.backward()
    assert model.trl_a.core.grad is not None
    assert model.trl_v.core.grad is not None

    model.eval()
    audio, video, _, text, *_ = _inputs(batch_size=5)
    with torch.no_grad():
        embeddings = model.get_embeddings(audio, video, text)
    assert all(embedding.shape[0] == 5 for embedding in embeddings)


def test_stft_ablation_flags_bypass_glp_and_lkc():
    params = _small_params()
    params.update(use_glp=False, use_lkc=False)
    model = MSTR(params, input_size_audio=16, input_size_video=16)

    assert not model.use_glp
    assert not model.use_lkc
    assert not model.SNNbranchaudio.use_glp
    model.eval()
    audio, video, _, text, *_ = _inputs()
    with torch.no_grad():
        outputs = model.get_embeddings(audio, video, text)
    assert all(torch.isfinite(output).all() for output in outputs)


def test_stft_temporal_features_are_normalized_and_lkc_is_residual():
    torch.manual_seed(7)
    params = _small_params()
    params["lkc_residual_scale"] = 0.2
    model = MSTR(params, input_size_audio=16, input_size_video=16).eval()
    audio, video, _, _, *_ = _inputs()

    with torch.no_grad():
        _, temporal_audio = model._encode_temporal_audio(audio)
        _, temporal_video = model._encode_temporal_video(video)
        phi_a = model.A_enc(audio)
        phi_v = model.V_enc(video)
        refined_a, refined_v = model._apply_lkc_residual(phi_a, phi_v)

    for temporal in (temporal_audio, temporal_video):
        torch.testing.assert_close(
            temporal.mean(dim=-1), torch.zeros(temporal.shape[0]),
            atol=1e-5, rtol=0.0)
        assert torch.isfinite(temporal).all()

    gate = params["lkc_residual_scale"] * model.lkc.alpha.detach()
    for base, refined in ((phi_a, refined_a), (phi_v, refined_v)):
        normalized_residual = (refined - base) / gate
        torch.testing.assert_close(
            normalized_residual.mean(dim=-1),
            torch.zeros(normalized_residual.shape[0]), atol=1e-5, rtol=0.0)
        assert torch.isfinite(normalized_residual).all()


def test_glp_amax_matches_max_value_and_keeps_gradients_finite():
    torch.manual_seed(5)
    glp = GlobalLocalPool()
    features = torch.randn(7, 13, requires_grad=True)

    output, context = glp(features, return_context=True)
    beta = glp.beta
    p_max = features.max(dim=-1, keepdim=True).values
    p_avg = features.mean(dim=-1, keepdim=True)
    expected_context = (
        0.5 * (p_max + p_avg) + beta * p_max + (1.0 - beta) * p_avg)
    expected_output = (1.0 + torch.sigmoid(expected_context)) * features

    torch.testing.assert_close(context, expected_context)
    torch.testing.assert_close(output, expected_output)
    output.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert torch.isfinite(glp.beta_logit.grad)


def test_contrastive_dataset_uses_wrapped_split_as_source_of_truth():
    class DatasetStub:
        dataset_split = "train"
        classes = [0, 1]
        targets = np.asarray([0, 0, 1, 1])
        all_data = {
            "audio": torch.randn(4, 2),
            "video": torch.randn(4, 2),
            "text": torch.randn(2, 300),
            "url": np.asarray(["a", "b", "c", "d"]),
        }

    wrapped = DatasetStub()
    dataset = ContrastiveDataset(wrapped)
    assert dataset.dataset_split == "train"
    wrapped.dataset_split = "train_val"
    assert dataset.dataset_split == "train_val"


class ModelImprovementsTest(unittest.TestCase):
    test_lif_has_hard_forward_and_sigmoid_surrogate_gradient = staticmethod(
        test_lif_has_hard_forward_and_sigmoid_surrogate_gradient
    )
    test_gradient_reverse_flips_only_the_backward_signal = staticmethod(
        test_gradient_reverse_flips_only_the_backward_signal
    )
    test_text_batchnorm_recalibration_uses_only_class_semantics = staticmethod(
        test_text_batchnorm_recalibration_uses_only_class_semantics
    )
    test_full_model_backward_reaches_every_parameter = staticmethod(
        test_full_model_backward_reaches_every_parameter
    )
    test_global_prototype_contrastive_loss_uses_nonpersistent_text_dictionary = staticmethod(
        test_global_prototype_contrastive_loss_uses_nonpersistent_text_dictionary
    )
    test_cross_modal_contrastive_loss_is_finite_and_reaches_snn_encoders = staticmethod(
        test_cross_modal_contrastive_loss_is_finite_and_reaches_snn_encoders
    )
    test_snn_temporal_consistency_is_train_only_and_reaches_snn = staticmethod(
        test_snn_temporal_consistency_is_train_only_and_reaches_snn
    )
    test_semantic_batch_hard_loss_is_finite_and_reaches_snn_encoders = staticmethod(
        test_semantic_batch_hard_loss_is_finite_and_reaches_snn_encoders
    )
    test_semantic_neighbor_rank_is_train_only_and_reaches_snn_encoders = staticmethod(
        test_semantic_neighbor_rank_is_train_only_and_reaches_snn_encoders
    )
    test_semantic_neighbor_rank_excludes_matching_class_from_far_set = staticmethod(
        test_semantic_neighbor_rank_excludes_matching_class_from_far_set
    )
    test_seen_teacher_distillation_is_masked_and_finite = staticmethod(
        test_seen_teacher_distillation_is_masked_and_finite
    )
    test_snn_activity_floor_is_opt_in_and_reaches_snn_parameters = staticmethod(
        test_snn_activity_floor_is_opt_in_and_reaches_snn_parameters
    )
    test_membrane_readout_preserves_hard_spike_diagnostics_and_is_opt_in = staticmethod(
        test_membrane_readout_preserves_hard_spike_diagnostics_and_is_opt_in
    )
    test_cross_modal_residual_gate_is_bounded_and_sample_adaptive = staticmethod(
        test_cross_modal_residual_gate_is_bounded_and_sample_adaptive
    )
    test_cross_modal_residual_is_opt_in_and_receives_gradients = staticmethod(
        test_cross_modal_residual_is_opt_in_and_receives_gradients
    )
    test_ahse_standardizer_uses_training_statistics_at_inference = staticmethod(
        test_ahse_standardizer_uses_training_statistics_at_inference
    )
    test_ahse_standardization_is_wired_into_mstr_without_changing_default_state = staticmethod(
        test_ahse_standardization_is_wired_into_mstr_without_changing_default_state
    )
    test_semantic_geometry_regularizer_preserves_text_relations_with_gradients = staticmethod(
        test_semantic_geometry_regularizer_preserves_text_relations_with_gradients
    )
    test_semantic_contrastive_loss_is_finite_and_reaches_modal_text_encoders = staticmethod(
        test_semantic_contrastive_loss_is_finite_and_reaches_modal_text_encoders
    )
    test_avla_contrastive_replaces_mstr_objective_and_reaches_snn_encoder = staticmethod(
        test_avla_contrastive_replaces_mstr_objective_and_reaches_snn_encoder
    )
    test_semantic_hard_negative_loss_uses_different_class_text_with_gradients = staticmethod(
        test_semantic_hard_negative_loss_uses_different_class_text_with_gradients
    )
    test_semantic_mixup_aligns_different_class_manifolds_with_gradients = staticmethod(
        test_semantic_mixup_aligns_different_class_manifolds_with_gradients
    )
    test_feature_mixup_runs_through_snn_stft_and_reaches_encoders = staticmethod(
        test_feature_mixup_runs_through_snn_stft_and_reaches_encoders
    )
    test_feature_debias_is_opt_in_and_trains_semantic_residual_paths = staticmethod(
        test_feature_debias_is_opt_in_and_trains_semantic_residual_paths
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
    test_snn_embeddings_are_invariant_to_batch_companions = staticmethod(
        test_snn_embeddings_are_invariant_to_batch_companions
    )
    test_legacy_batch_dth_restores_batch_companion_dependence = staticmethod(
        test_legacy_batch_dth_restores_batch_companion_dependence
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
    test_stft_vector_trl_is_opt_in_sample_dependent_and_trainable = staticmethod(
        test_stft_vector_trl_is_opt_in_sample_dependent_and_trainable
    )
    test_spatial_reliability_gate_is_bounded_and_sample_adaptive = staticmethod(
        test_spatial_reliability_gate_is_bounded_and_sample_adaptive
    )
    test_spatial_reliability_endpoints_bypass_or_restore_spatial_branch = staticmethod(
        test_spatial_reliability_endpoints_bypass_or_restore_spatial_branch
    )
    test_stft_spatial_reliability_gate_reaches_full_training_objective = staticmethod(
        test_stft_spatial_reliability_gate_reaches_full_training_objective
    )
    test_frozen_residual_backbone_keeps_batchnorm_statistics_fixed = staticmethod(
        test_frozen_residual_backbone_keeps_batchnorm_statistics_fixed
    )
    test_mstr_baseline_variants_match_released_and_paper_attention_layouts = staticmethod(
        test_mstr_baseline_variants_match_released_and_paper_attention_layouts
    )
    test_paper_mstr_baseline_supports_dynamic_batches_and_backward = staticmethod(
        test_paper_mstr_baseline_supports_dynamic_batches_and_backward
    )
    test_stft_ablation_flags_bypass_glp_and_lkc = staticmethod(
        test_stft_ablation_flags_bypass_glp_and_lkc
    )
    test_stft_temporal_features_are_normalized_and_lkc_is_residual = staticmethod(
        test_stft_temporal_features_are_normalized_and_lkc_is_residual
    )
    test_glp_amax_matches_max_value_and_keeps_gradients_finite = staticmethod(
        test_glp_amax_matches_max_value_and_keeps_gradients_finite
    )
    test_contrastive_dataset_uses_wrapped_split_as_source_of_truth = staticmethod(
        test_contrastive_dataset_uses_wrapped_split_as_source_of_truth
    )


if __name__ == "__main__":
    unittest.main()
