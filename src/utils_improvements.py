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

def get_model_params(lr, first_additional_triplet, second_additional_triplet,  reg_loss, additional_triplets_loss, dropout_encoder, dropout_decoder, additional_dropout, encoder_hidden_size, decoder_hidden_size, depth_transformer, momentum, snn_T=10, trl_rank=400, snn_tau=2.0, lkc_n_slots=4, lkc_n_heads=8, tucker_rank=60, stft_dim=512, fusion_mode="stft", vector_trl_rank=64, trl_gate_scale=0.25, backbone_lr_scale=1.0):
    # Model parameters
    params_model = dict()
    params_model['dim_out'] = 64
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
    params_model['lkc_n_slots']=lkc_n_slots
    params_model['lkc_n_heads']=lkc_n_heads
    params_model['tucker_rank']=tucker_rank
    params_model['stft_dim']=stft_dim
    params_model['fusion_mode']=fusion_mode
    params_model['vector_trl_rank']=vector_trl_rank
    params_model['trl_gate_scale']=trl_gate_scale
    params_model['backbone_lr_scale']=backbone_lr_scale
    return params_model
