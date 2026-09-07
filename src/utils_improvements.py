#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import socket
import random
import itertools
import numpy as np
import multiprocessing
import configparser as cp
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score

import torch

np.random.seed(0)

def get_model_params(lr, first_additional_triplet, second_additional_triplet,  reg_loss, additional_triplets_loss, dropout_encoder, dropout_decoder, additional_dropout, encoder_hidden_size, decoder_hidden_size, depth_transformer, momentum, snn_T=10, trl_rank=400, snn_tau=2.0, snn_activity_floor_weight=0.0, snn_min_spike_rate=0.05, snn_membrane_readout_scale=0.0, lkc_n_slots=4, lkc_n_heads=8, tucker_rank=60, stft_dim=512, fusion_mode="stft", vector_trl_rank=64, stft_vector_trl=False, stft_spatial_reliability_gate=False, trl_gate_scale=0.25, backbone_lr_scale=1.0, use_glp=True, use_lkc=True, legacy_batch_dth=False, lkc_residual_scale=0.2, ahse_standardize=False, semantic_geometry_weight=0.0, cross_modal_residual=False, cross_modal_residual_scale=0.2, semantic_contrastive_weight=0.0, semantic_contrastive_temperature=0.1, pseudo_unseen_weight=0.0, pseudo_unseen_temperature=0.15, pseudo_unseen_class_fraction=0.5, pseudo_unseen_min_classes=2, snn_temporal_consistency_weight=0.0, snn_temporal_view_fraction=0.25, temporal_quality_alignment_weight=0.0, cross_modal_contrastive_weight=0.0, cross_modal_contrastive_temperature=0.1, avla_contrastive_only=False, avla_temperature=0.1, global_prototype_contrastive_weight=0.0, global_prototype_contrastive_temperature=0.1, semantic_hard_negative_weight=0.0, semantic_hard_negative_margin=0.1, semantic_batch_hard_weight=0.0, semantic_batch_hard_margin=0.1, semantic_batch_hard_neighbors=5, semantic_mixup_weight=0.0, semantic_mixup_alpha=1.0, feature_mixup_weight=0.0, feature_mixup_alpha=0.2, feature_debias_weight=0.0, feature_debias_temperature=0.1, text_projection_norm="batchnorm", semantic_neighbor_rank_weight=0.0, semantic_neighbor_rank_margin=0.05, semantic_neighbor_rank_neighbors=5, text_embedding_size=300):
    # Model parameters
    params_model = dict()
    params_model['dim_out'] = 64
    params_model['text_embedding_size'] = int(text_embedding_size)
    params_model['lr'] = lr
    if encoder_hidden_size==0:
        encoder_hidden_size=None
    if decoder_hidden_size==0:
        decoder_hidden_size=None
    params_model['first_additional_triplet'] = first_additional_triplet
    params_model['second_additional_triplet'] = second_additional_triplet
    params_model['additional_triplets_loss']=additional_triplets_loss
    params_model['additional_dropout'] = additional_dropout
    params_model['reg_loss']=reg_loss
    params_model['depth_transformer']=depth_transformer
    params_model['dropout_encoder']=dropout_encoder
    params_model['dropout_decoder']=dropout_decoder
    params_model['encoder_hidden_size']=encoder_hidden_size
    params_model['decoder_hidden_size']=decoder_hidden_size
    params_model['momentum']=momentum
    params_model['snn_T']=snn_T
    params_model['trl_rank']=trl_rank
    # --- STFT upgrade hyperparameters ---
    params_model['snn_tau']=snn_tau
    params_model['snn_activity_floor_weight']=snn_activity_floor_weight
    params_model['snn_min_spike_rate']=snn_min_spike_rate
    params_model['snn_membrane_readout_scale']=snn_membrane_readout_scale
    params_model['legacy_batch_dth']=legacy_batch_dth
    params_model['lkc_n_slots']=lkc_n_slots
    params_model['lkc_n_heads']=lkc_n_heads
    params_model['tucker_rank']=tucker_rank
    params_model['stft_dim']=stft_dim
    params_model['fusion_mode']=fusion_mode
    params_model['vector_trl_rank']=vector_trl_rank
    params_model['stft_vector_trl']=stft_vector_trl
    params_model['stft_spatial_reliability_gate']=stft_spatial_reliability_gate
    params_model['trl_gate_scale']=trl_gate_scale
    params_model['backbone_lr_scale']=backbone_lr_scale
    params_model['use_glp']=use_glp
    params_model['use_lkc']=use_lkc
    # Configuration only: no new state-dict key is introduced.
    params_model['lkc_residual_scale']=lkc_residual_scale
    params_model['ahse_standardize']=ahse_standardize
    params_model['semantic_geometry_weight']=semantic_geometry_weight
    params_model['cross_modal_residual']=cross_modal_residual
    params_model['cross_modal_residual_scale']=cross_modal_residual_scale
    params_model['semantic_contrastive_weight']=semantic_contrastive_weight
    params_model['semantic_contrastive_temperature']=semantic_contrastive_temperature
    params_model['pseudo_unseen_weight']=pseudo_unseen_weight
    params_model['pseudo_unseen_temperature']=pseudo_unseen_temperature
    params_model['pseudo_unseen_class_fraction']=pseudo_unseen_class_fraction
    params_model['pseudo_unseen_min_classes']=pseudo_unseen_min_classes
    params_model['snn_temporal_consistency_weight']=snn_temporal_consistency_weight
    params_model['snn_temporal_view_fraction']=snn_temporal_view_fraction
    params_model['temporal_quality_alignment_weight']=temporal_quality_alignment_weight
    params_model['cross_modal_contrastive_weight']=cross_modal_contrastive_weight
    params_model['cross_modal_contrastive_temperature']=cross_modal_contrastive_temperature
    params_model['avla_contrastive_only']=avla_contrastive_only
    params_model['avla_temperature']=avla_temperature
    params_model['global_prototype_contrastive_weight']=global_prototype_contrastive_weight
    params_model['global_prototype_contrastive_temperature']=global_prototype_contrastive_temperature
    params_model['semantic_hard_negative_weight']=semantic_hard_negative_weight
    params_model['semantic_hard_negative_margin']=semantic_hard_negative_margin
    params_model['semantic_batch_hard_weight']=semantic_batch_hard_weight
    params_model['semantic_batch_hard_margin']=semantic_batch_hard_margin
    params_model['semantic_batch_hard_neighbors']=semantic_batch_hard_neighbors
    params_model['semantic_neighbor_rank_weight']=semantic_neighbor_rank_weight
    params_model['semantic_neighbor_rank_margin']=semantic_neighbor_rank_margin
    params_model['semantic_neighbor_rank_neighbors']=semantic_neighbor_rank_neighbors
    params_model['semantic_mixup_weight']=semantic_mixup_weight
    params_model['semantic_mixup_alpha']=semantic_mixup_alpha
    params_model['feature_mixup_weight']=feature_mixup_weight
    params_model['feature_mixup_alpha']=feature_mixup_alpha
    params_model['feature_debias_weight']=feature_debias_weight
    params_model['feature_debias_temperature']=feature_debias_temperature
    params_model['text_projection_norm']=text_projection_norm
    return params_model
